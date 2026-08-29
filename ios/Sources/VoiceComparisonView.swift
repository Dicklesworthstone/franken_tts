// Voice Lab: a fair, device-local audition of one exact excerpt across every
// built-in and personal voice. Rust shares the text-side plan; each card is still
// the output of a real speaker-conditioned generation.

import AVFoundation
import Observation
import SwiftUI
import UIKit

struct VoiceComparisonCandidate: Identifiable, Sendable {
    let id: String
    let selectionID: String
    let name: String
    let character: String
    let speaker: [Float]
    let personal: Bool
}

struct VoiceComparisonResult: Sendable {
    let url: URL
    let duration: TimeInterval
    /// Compact, sample-derived time/frequency analysis used by the same interactive
    /// playback instrument as the main forge. Keeping this instead of every voice's
    /// full PCM avoids retaining hundreds of megabytes beside the resident model.
    let signalAnalysis: SignalAnalysis
    let profile: SynthesisProfile?
}

fileprivate enum VoiceComparisonStatus: Equatable {
    case waiting
    case forging
    case mastering
    case ready
    case failed(String)
}

/// Native synthesis can report many semantic-frame callbacks per second. Keeping
/// only the newest one lets the engine run freely while SwiftUI renders at a calm,
/// predictable cadence instead of rebuilding the whole voice grid for every frame.
private final class VoiceComparisonProgressRelay: @unchecked Sendable {
    struct Pending: Sendable {
        let index: Int
        let event: EngineProgress
        let runID: UUID
    }

    private let lock = NSLock()
    private var pending: Pending?

    func offer(index: Int, event: EngineProgress, runID: UUID) {
        lock.lock()
        pending = Pending(index: index, event: event, runID: runID)
        lock.unlock()
    }

    func take() -> Pending? {
        lock.lock()
        defer { lock.unlock() }
        let value = pending
        pending = nil
        return value
    }

    func clear() {
        lock.lock()
        pending = nil
        lock.unlock()
    }
}

@MainActor
@Observable
final class VoiceComparisonSession {
    private static let favoritesKey = "voiceLab.favoriteVoiceIDs.v1"

    let model: LabModel
    let candidates: [VoiceComparisonCandidate]
    var excerpt: String
    fileprivate var status: [String: VoiceComparisonStatus] = [:]
    var results: [String: VoiceComparisonResult] = [:]
    var favorites: Set<String>
    var isRunning = false
    var isStopping = false
    var activeIndex: Int?
    var activeStage = "Ready for an all-voice audition"
    var activeFraction = 0.0
    var completedCount = 0
    var settledCount = 0
    var elapsed: TimeInterval = 0
    var estimatedRemaining: Int?
    var errorMessage: String?
    var playingID: String?
    var playbackFraction = 0.0
    var playbackRevision = 0

    private var runTask: Task<Void, Never>?
    private var tickerTask: Task<Void, Never>?
    private var progressTask: Task<Void, Never>?
    private var runID = UUID()
    private var observedVoiceSeconds = 0.0
    private var player: AVAudioPlayer?
    private let progressRelay = VoiceComparisonProgressRelay()

    init(model: LabModel) {
        self.model = model
        excerpt = String(model.text.prefix(UtteranceChunker.maximumChunkCharacters))
        var built: [VoiceComparisonCandidate] = []
        for preset in model.presets {
            guard let vector = try? Engine.presetVector(named: preset.name) else { continue }
            built.append(
                VoiceComparisonCandidate(
                    id: "preset:\(preset.name)",
                    selectionID: preset.name,
                    name: preset.name.capitalized,
                    character: preset.character,
                    speaker: vector,
                    personal: false
                ))
        }
        built.append(contentsOf: model.library.voices.map { voice in
            VoiceComparisonCandidate(
                id: "voice:\(voice.id.uuidString)",
                selectionID: "voice:\(voice.id.uuidString)",
                name: voice.name,
                character: "Your private, locally enrolled voice",
                speaker: voice.vector,
                personal: true
            )
        })
        candidates = built
        favorites = Set(
            UserDefaults.standard.stringArray(forKey: Self.favoritesKey) ?? [])
        status = Dictionary(uniqueKeysWithValues: built.map { ($0.id, .waiting) })

        #if DEBUG
            if ProcessInfo.processInfo.environment["FTTS_VOICE_LAB_FIXTURE"] == "1" {
                installVisualTestFixture()
            }
        #endif
    }

    var canStart: Bool {
        !isRunning
            && !candidates.isEmpty
            && model.canCompareVoices
            && !excerpt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var activeVoiceName: String? {
        guard let activeIndex, candidates.indices.contains(activeIndex) else { return nil }
        return candidates[activeIndex].name
    }

    func start() {
        guard canStart else { return }
        stopPlayback()
        runTask?.cancel()
        tickerTask?.cancel()
        progressTask?.cancel()
        progressRelay.clear()
        removeTemporaryResults()
        runID = UUID()
        let thisRun = runID
        errorMessage = nil
        status = Dictionary(uniqueKeysWithValues: candidates.map { ($0.id, .waiting) })
        isRunning = true
        isStopping = false
        activeIndex = nil
        activeStage = model.isEngineWarm ? "Binding one shared text plan" : "Waking the voice core"
        activeFraction = 0
        completedCount = 0
        settledCount = 0
        elapsed = 0
        estimatedRemaining = nil
        observedVoiceSeconds = 0
        model.isComparingVoices = true

        let started = Date()
        tickerTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(250))
                guard let self, self.runID == thisRun, self.isRunning else { return }
                self.elapsed = Date().timeIntervalSince(started)
            }
        }

        progressTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(80))
                guard let self, self.runID == thisRun, self.isRunning else { return }
                if let progress = self.progressRelay.take() {
                    self.receiveProgress(
                        index: progress.index,
                        event: progress.event,
                        runID: progress.runID
                    )
                }
            }
        }

        let text = excerpt
        let inputs = candidates.map { MultiVoiceSynthesisInput(speaker: $0.speaker) }
        let seed = model.seed
        let progressRelay = progressRelay
        UIApplication.shared.isIdleTimerDisabled = true
        runTask = Task { [weak self] in
            guard let self else { return }
            defer {
                UIApplication.shared.isIdleTimerDisabled = false
                self.tickerTask?.cancel()
                self.tickerTask = nil
                self.progressTask?.cancel()
                self.progressTask = nil
                self.progressRelay.clear()
                self.isRunning = false
                self.isStopping = false
                self.model.isComparingVoices = false
                self.activeIndex = nil
                self.activeFraction = 0
                if self.completedCount == self.candidates.count {
                    self.activeStage = "All voices forged — tap to compare"
                    self.estimatedRemaining = nil
                }
            }
            do {
                try Task.checkCancellation()
                if await !self.model.engine.isLoaded {
                    self.model.isLoadingModel = true
                    do {
                        try await self.model.engine.load(
                            modelDirectory: self.model.store.modelDirectory)
                        self.model.isEngineWarm = true
                        self.model.isLoadingModel = false
                    } catch {
                        self.model.isLoadingModel = false
                        throw error
                    }
                }
                try Task.checkCancellation()
                try await self.model.engine.synthesizeMany(
                    text: text,
                    voices: inputs,
                    seed: seed,
                    onProgress: { index, event in
                        progressRelay.offer(index: index, event: event, runID: thisRun)
                    },
                    onVoice: { [weak self] index, output in
                        Task { @MainActor in
                            self?.receiveVoice(index: index, output: output, runID: thisRun)
                        }
                    }
                )
                // Native delivery is complete here, but each card's mastering and
                // atomic WAV write happens off the main actor. Retain run ownership
                // until every delivered voice is genuinely playable (or failed).
                let masteringDeadline = Date().addingTimeInterval(60)
                while self.settledCount < self.candidates.count {
                    try Task.checkCancellation()
                    guard Date() < masteringDeadline else {
                        throw EngineError.native(
                            "audio mastering did not settle after the voice engine finished")
                    }
                    try await Task.sleep(for: .milliseconds(25))
                }
            } catch EngineError.cancelled {
                if self.runID == thisRun { self.activeStage = "Voice Lab stopped" }
            } catch is CancellationError {
                if self.runID == thisRun { self.activeStage = "Voice Lab stopped" }
            } catch {
                guard self.runID == thisRun else { return }
                self.errorMessage = error.localizedDescription
                self.activeStage = "Voice Lab needs attention"
                if let activeIndex, self.candidates.indices.contains(activeIndex) {
                    self.status[self.candidates[activeIndex].id] = .failed(
                        error.localizedDescription)
                }
            }
        }
    }

    func stop() {
        guard isRunning else { return }
        isStopping = true
        activeStage = "Settling the current frame…"
        model.engine.cancelCurrentWork()
        runTask?.cancel()
        // Fence progress and mastering callbacks immediately. A native PCM callback
        // already in flight may still finish its file, but it will delete that stale
        // file instead of repopulating a cancelled session.
        runID = UUID()
    }

    func cancelForDismissal() {
        if isRunning { stop() }
        stopPlayback()
        removeTemporaryResults()
    }

    func toggleFavorite(_ candidate: VoiceComparisonCandidate) {
        if favorites.contains(candidate.id) {
            favorites.remove(candidate.id)
        } else {
            favorites.insert(candidate.id)
        }
        UserDefaults.standard.set(Array(favorites).sorted(), forKey: Self.favoritesKey)
    }

    func useVoice(_ candidate: VoiceComparisonCandidate) {
        model.selectedVoice = candidate.selectionID
    }

    func togglePlayback(_ candidate: VoiceComparisonCandidate) {
        guard let result = results[candidate.id] else { return }
        if playingID == candidate.id, let player {
            if player.isPlaying {
                player.pause()
            } else {
                if player.duration > 0, player.currentTime >= player.duration - 0.05 {
                    player.currentTime = 0
                    playbackFraction = 0
                }
                player.play()
            }
            playbackRevision &+= 1
            return
        }
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback)
            player = try AVAudioPlayer(contentsOf: result.url)
            playingID = candidate.id
            playbackFraction = 0
            player?.play()
            playbackRevision &+= 1
        } catch {
            errorMessage = "Could not play \(candidate.name): \(error.localizedDescription)"
        }
    }

    func playbackPlayer(for candidate: VoiceComparisonCandidate) -> AVAudioPlayer? {
        playingID == candidate.id ? player : nil
    }

    /// Seek one card without starting audio under the user's finger. The shared signal
    /// view calls `finishScrubbing` on release, which resumes an existing preview or
    /// starts the newly selected one from the exact touched position.
    func seekPlayback(_ candidate: VoiceComparisonCandidate, to progress: Double) {
        guard let result = results[candidate.id] else { return }
        do {
            if playingID != candidate.id || player == nil {
                try AVAudioSession.sharedInstance().setCategory(.playback)
                player = try AVAudioPlayer(contentsOf: result.url)
                playingID = candidate.id
            }
            guard let player, player.duration > 0 else { return }
            let fraction = min(1, max(0, progress))
            player.currentTime = player.duration * fraction
            playbackFraction = fraction
            playbackRevision &+= 1
        } catch {
            errorMessage = "Could not seek \(candidate.name): \(error.localizedDescription)"
        }
    }

    func finishScrubbing(_ candidate: VoiceComparisonCandidate) {
        guard playingID == candidate.id, let player else { return }
        if player.duration > 0, player.currentTime >= player.duration - 0.05 {
            player.currentTime = 0
            playbackFraction = 0
        }
        player.play()
        playbackRevision &+= 1
    }

    func refreshPlayback() {
        guard let player, playingID != nil else { return }
        playbackFraction = player.duration > 0
            ? min(1, max(0, player.currentTime / player.duration))
            : 0
        if !player.isPlaying, player.currentTime >= player.duration - 0.05 {
            playbackFraction = 1
        }
    }

    func isPlaying(_ candidate: VoiceComparisonCandidate) -> Bool {
        playingID == candidate.id && player?.isPlaying == true
    }

    private func stopPlayback() {
        player?.stop()
        player = nil
        playingID = nil
        playbackFraction = 0
        playbackRevision &+= 1
    }

    private func removeTemporaryResults() {
        let urls = results.values.map(\.url)
        results.removeAll(keepingCapacity: false)
        for url in urls {
            try? FileManager.default.removeItem(at: url)
        }
    }

    private func receiveProgress(index: Int, event: EngineProgress, runID: UUID) {
        guard self.runID == runID, isRunning, candidates.indices.contains(index) else { return }
        if activeIndex != index {
            if let old = activeIndex,
               candidates.indices.contains(old),
               status[candidates[old].id] == .forging
            {
                status[candidates[old].id] = .mastering
            }
            activeIndex = index
            status[candidates[index].id] = .forging
            activeFraction = 0
        }
        activeStage = event.stage.shortLabel
        if event.kind == .unit,
           (event.stage == .frames || event.stage == .codec),
           event.total > 0
        {
            activeFraction = min(0.99, Double(event.current) / Double(event.total))
        }
    }

    private func receiveVoice(index: Int, output: SynthesisOutput, runID: UUID) {
        guard self.runID == runID, candidates.indices.contains(index) else { return }
        let candidate = candidates[index]
        status[candidate.id] = .mastering
        completedCount = max(completedCount, index + 1)
        if let profile = output.profile {
            observedVoiceSeconds += profile.totalMs / 1_000
            let average = observedVoiceSeconds / Double(max(1, completedCount))
            estimatedRemaining = Int(ceil(average * Double(candidates.count - completedCount)))
        }

        Task { [weak self] in
            let finished = await Task.detached(priority: .userInitiated) {
                let mastered = SpeechMastering.process(output.pcm)
                let fileName = Self.safeFileComponent(candidate.name)
                let url = FileManager.default.temporaryDirectory.appendingPathComponent(
                    "FrankenTTS-\(fileName)-\(index + 1)-\(runID.uuidString.prefix(8)).wav")
                try WavWriter.data(from: mastered).write(to: url, options: .atomic)
                return VoiceComparisonResult(
                    url: url,
                    duration: Double(mastered.count) / Double(WavWriter.sampleRate),
                    signalAnalysis: SignalAnalysis(samples: mastered),
                    profile: output.profile
                )
            }.result
            guard let self else {
                if case .success(let result) = finished {
                    try? FileManager.default.removeItem(at: result.url)
                }
                return
            }
            guard self.runID == runID else {
                if case .success(let result) = finished {
                    try? FileManager.default.removeItem(at: result.url)
                }
                return
            }
            switch finished {
            case .success(let result):
                self.results[candidate.id] = result
                self.status[candidate.id] = .ready
            case .failure(let error):
                self.status[candidate.id] = .failed(error.localizedDescription)
            }
            self.settledCount += 1
        }
    }

    nonisolated private static func safeFileComponent(_ name: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
        let scalars = name.unicodeScalars.map { allowed.contains($0) ? Character(String($0)) : "-" }
        let cleaned = String(scalars).trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return cleaned.isEmpty ? "Voice" : String(cleaned.prefix(48))
    }

    #if DEBUG
        /// Gives UI tests all important card states without pretending the simulator
        /// can run a damaged multi-gigabyte model installation. This path is compiled
        /// out of release builds and requires an explicit test environment variable.
        private func installVisualTestFixture() {
            guard !candidates.isEmpty else { return }
            isRunning = true
            activeIndex = 0
            activeStage = "Growing semantic voice frames"
            activeFraction = 0.42
            elapsed = 7
            estimatedRemaining = 19
            status[candidates[0].id] = .forging

            if candidates.indices.contains(1) {
                let samples = (0..<(WavWriter.sampleRate * 5)).map { index -> Float in
                    let time = Float(index) / Float(WavWriter.sampleRate)
                    let carrier = sin(2 * .pi * 165 * time) * 0.16
                    let overtone = sin(2 * .pi * 330 * time) * 0.07
                    let envelope = 0.38 + 0.62 * abs(sin(2 * .pi * 1.7 * time))
                    return (carrier + overtone) * envelope
                }
                let placeholder = FileManager.default.temporaryDirectory
                    .appendingPathComponent("FrankenTTS-Voice-Lab-visual-fixture.wav")
                try? WavWriter.data(from: samples).write(to: placeholder, options: .atomic)
                results[candidates[1].id] = VoiceComparisonResult(
                    url: placeholder,
                    duration: Double(samples.count) / Double(WavWriter.sampleRate),
                    signalAnalysis: SignalAnalysis(samples: samples),
                    profile: nil
                )
                status[candidates[1].id] = .ready
                completedCount = 1
                settledCount = 1
            }
            if candidates.indices.contains(2) {
                status[candidates[2].id] = .mastering
            }
        }
    #endif
}

struct VoiceComparisonView: View {
    @State private var session: VoiceComparisonSession
    @State private var showFavoritesOnly = false
    @State private var showLeaveConfirmation = false
    let dismiss: () -> Void

    init(model: LabModel, dismiss: @escaping () -> Void) {
        _session = State(initialValue: VoiceComparisonSession(model: model))
        self.dismiss = dismiss
    }

    private var visibleCandidates: [VoiceComparisonCandidate] {
        showFavoritesOnly
            ? session.candidates.filter { session.favorites.contains($0.id) }
            : session.candidates
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                hero
                auditionEditor
                runStatus
                resultGrid
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 20)
            .frame(maxWidth: 1_080)
            .frame(maxWidth: .infinity)
        }
        .scrollIndicators(.hidden)
        .background(LaboratoryBackground())
        .navigationTitle("Voice Lab")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button("Done") {
                    if session.isRunning {
                        showLeaveConfirmation = true
                    } else {
                        dismiss()
                    }
                }
                .accessibilityHint(
                    session.isRunning
                        ? "Asks before stopping the active all-voice generation"
                        : "Closes Voice Lab"
                )
            }
        }
        .interactiveDismissDisabled(session.isRunning)
        .alert("Voice Lab is still generating", isPresented: $showLeaveConfirmation) {
            Button("Keep generating", role: .cancel) {}
            Button("Stop and leave", role: .destructive) {
                session.stop()
                dismiss()
            }
        } message: {
            Text(
                "Leaving now stops the current run. Finished voice previews from this audition will be discarded."
            )
        }
        .onDisappear { session.cancelForDismissal() }
        .onReceive(Timer.publish(every: 0.35, on: .main, in: .common).autoconnect()) { _ in
            session.refreshPlayback()
        }
    }

    private var hero: some View {
        LabPanel {
            HStack(alignment: .top, spacing: 16) {
                VoiceLabPulse(active: session.isRunning)
                    .frame(width: 64, height: 64)
                VStack(alignment: .leading, spacing: 6) {
                    LabLabel(text: "One text · every voice")
                    Text("Hear the whole cast")
                        .font(.system(size: Lab.typeSize(24), weight: .black, design: .rounded))
                        .foregroundStyle(Lab.textPrimary)
                    Text(
                        "FrankenTTS prepares this excerpt once, then gives every voice the same words and seed. Favorite the voices that fit; nothing leaves this device."
                    )
                    .font(.system(size: Lab.typeSize(13)))
                    .foregroundStyle(Lab.textSecondary)
                }
                Spacer(minLength: 0)
            }
        }
    }

    private var auditionEditor: some View {
        LabPanel {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    LabLabel(text: "Audition excerpt")
                    Spacer()
                    Text("\(session.excerpt.count) / \(UtteranceChunker.maximumChunkCharacters)")
                        .font(.caption.monospaced())
                        .foregroundStyle(Lab.textSecondary)
                }
                TextEditor(
                    text: Binding(
                        get: { session.excerpt },
                        set: {
                            session.excerpt = String(
                                $0.prefix(UtteranceChunker.maximumChunkCharacters))
                        }
                    )
                )
                .scrollContentBackground(.hidden)
                .font(.system(size: Lab.typeSize(16)))
                .foregroundStyle(Lab.textPrimary)
                .padding(8)
                .frame(minHeight: 112, maxHeight: 180)
                .background(Color.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 12))
                .overlay(RoundedRectangle(cornerRadius: 12).stroke(Lab.emerald.opacity(0.24)))
                .disabled(session.isRunning)
                Text(
                    "Voice Lab is a focused audition, so it uses one exact model-sized excerpt. After choosing a favorite, the main forge can speak the complete 50,000-character document."
                )
                .font(.caption)
                .foregroundStyle(Lab.textSecondary)

                HStack(spacing: 10) {
                    Button {
                        session.start()
                    } label: {
                        Label(
                            session.isRunning ? "Forging the cast" : "Forge all \(session.candidates.count) voices",
                            systemImage: "person.3.sequence.fill"
                        )
                    }
                    .buttonStyle(PrimaryButtonStyle())
                    .disabled(!session.canStart)

                    if session.isRunning {
                        Button {
                            session.stop()
                        } label: {
                            Label(session.isStopping ? "Stopping" : "Stop", systemImage: "stop.fill")
                        }
                        .buttonStyle(GhostButtonStyle(tint: Lab.danger))
                        .disabled(session.isStopping)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var runStatus: some View {
        if session.isRunning || session.completedCount > 0 || session.errorMessage != nil {
            LabPanel {
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        VStack(alignment: .leading, spacing: 3) {
                            Text(session.activeStage)
                                .font(.headline)
                                .foregroundStyle(Lab.textPrimary)
                            if let voice = session.activeVoiceName {
                                Text("Now forging \(voice)")
                                    .font(.caption.monospaced())
                                    .foregroundStyle(Lab.emerald)
                            }
                        }
                        Spacer()
                        Text("\(session.completedCount) / \(session.candidates.count)")
                            .font(.headline.monospaced())
                            .foregroundStyle(Lab.textPrimary)
                    }
                    ProgressView(
                        value: Double(session.completedCount) + session.activeFraction,
                        total: Double(max(1, session.candidates.count))
                    )
                    .tint(Lab.emerald)
                    HStack {
                        Text("\(Int(session.elapsed))s elapsed")
                        Spacer()
                        if let remaining = session.estimatedRemaining, session.isRunning {
                            Text("about \(remaining)s remaining")
                        }
                    }
                    .font(.caption.monospaced())
                    .foregroundStyle(Lab.textSecondary)
                    if session.isRunning {
                        Label(
                            "Generation protected — Voice Lab will confirm before leaving",
                            systemImage: "lock.fill"
                        )
                        .font(.caption)
                        .foregroundStyle(Lab.emerald.opacity(0.88))
                    }
                    if let error = session.errorMessage {
                        Label(error, systemImage: "exclamationmark.triangle.fill")
                            .font(.caption)
                            .foregroundStyle(Lab.danger)
                    }
                }
            }
        }
    }

    private var resultGrid: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                LabLabel(text: "The cast")
                Spacer()
                Picker("Voice filter", selection: $showFavoritesOnly) {
                    Text("All \(session.candidates.count)").tag(false)
                    Text("Favorites \(session.favorites.count)").tag(true)
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 260)
            }
            if visibleCandidates.isEmpty {
                ContentUnavailableView(
                    "No favorites yet",
                    systemImage: "heart",
                    description: Text("Tap the heart on a voice you want to keep.")
                )
                .foregroundStyle(Lab.textSecondary)
            } else {
                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 290), spacing: 12)],
                    spacing: 12
                ) {
                    ForEach(visibleCandidates) { candidate in
                        VoiceComparisonCard(candidate: candidate, session: session)
                    }
                }
            }
        }
    }
}

private struct VoiceComparisonCard: View {
    let candidate: VoiceComparisonCandidate
    let session: VoiceComparisonSession

    private var status: VoiceComparisonStatus {
        session.status[candidate.id] ?? .waiting
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                VoiceComparisonOrb(candidate: candidate, active: status == .forging)
                    .frame(width: 48, height: 48)
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Text(candidate.name)
                            .font(.headline)
                            .foregroundStyle(Lab.textPrimary)
                        if candidate.personal {
                            Text("YOURS")
                                .font(.caption2.bold().monospaced())
                                .foregroundStyle(Lab.violet)
                        }
                    }
                    Text(candidate.character)
                        .font(.caption)
                        .foregroundStyle(Lab.textSecondary)
                        .lineLimit(2)
                        .frame(minHeight: 30, alignment: .topLeading)
                }
                Spacer()
                Button { session.toggleFavorite(candidate) } label: {
                    Image(systemName: session.favorites.contains(candidate.id) ? "heart.fill" : "heart")
                        .foregroundStyle(
                            session.favorites.contains(candidate.id) ? Lab.danger : Lab.textSecondary)
                        .frame(width: 44, height: 44)
                }
                .buttonStyle(.plain)
                .accessibilityLabel(
                    session.favorites.contains(candidate.id) ? "Remove favorite" : "Favorite \(candidate.name)")
            }

            Group {
                if let result = session.results[candidate.id] {
                    VStack(spacing: 8) {
                        PlaybackSignalView(
                            samples: [],
                            player: session.playbackPlayer(for: candidate),
                            analysisID: result.url.lastPathComponent,
                            refreshToken: session.playbackRevision,
                            preparedAnalysis: result.signalAnalysis,
                            resumesPlaybackAfterSeek: true,
                            onSeekFinished: { session.finishScrubbing(candidate) }
                        ) { progress in
                            session.seekPlayback(candidate, to: progress)
                        }
                        .accessibilityIdentifier("voice-lab-signal-\(candidate.id)")
                        .frame(maxWidth: .infinity)
                        .frame(height: 96)
                        HStack(spacing: 8) {
                            Button { session.togglePlayback(candidate) } label: {
                                Label(
                                    session.isPlaying(candidate) ? "Pause" : "Play",
                                    systemImage: session.isPlaying(candidate)
                                        ? "pause.fill" : "play.fill"
                                )
                            }
                            .buttonStyle(GhostButtonStyle(tint: Lab.emerald))
                            Button { session.useVoice(candidate) } label: {
                                Label("Use voice", systemImage: "checkmark.circle")
                            }
                            .buttonStyle(GhostButtonStyle(
                                tint: session.model.selectedVoice == candidate.selectionID
                                    ? Lab.emerald : Lab.textSecondary))
                            ShareLink(item: result.url) {
                                Image(systemName: "square.and.arrow.up")
                            }
                            .buttonStyle(GhostButtonStyle())
                            Spacer()
                            Text(String(format: "%.1fs", result.duration))
                                .font(.caption.monospaced())
                                .foregroundStyle(Lab.textSecondary)
                        }
                    }
                } else {
                    HStack(spacing: 8) {
                        switch status {
                        case .waiting:
                            Image(systemName: "circle.dotted")
                            Text("Waiting in the lab")
                        case .forging:
                            ProgressView().tint(Lab.emerald).controlSize(.small)
                            Text("Speaker-conditioned generation")
                        case .mastering:
                            ProgressView().tint(Lab.cyan).controlSize(.small)
                            Text("Leveling the finished audio")
                        case .ready:
                            Image(systemName: "checkmark.circle.fill")
                            Text("Ready")
                        case .failed(let message):
                            Image(systemName: "exclamationmark.triangle.fill")
                            Text(message).lineLimit(2)
                        }
                    }
                    .font(.caption.monospaced())
                    .foregroundStyle(statusColor)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
                }
            }
            // Every card reserves the result controls from the start. As earlier
            // voices finish, later cards no longer jump dramatically up and down.
            .frame(height: 150, alignment: .top)
        }
        .frame(minHeight: 218, alignment: .top)
        .padding(15)
        .background(Lab.panelStrong, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .strokeBorder(status == .forging ? Lab.emerald.opacity(0.8) : Lab.stroke, lineWidth: 1)
        }
    }

    private var statusColor: Color {
        switch status {
        case .failed: Lab.danger
        case .forging: Lab.emerald
        case .mastering: Lab.cyan
        default: Lab.textSecondary
        }
    }
}

private struct VoiceComparisonOrb: View {
    let candidate: VoiceComparisonCandidate
    let active: Bool

    var body: some View {
        ZStack {
            Circle().fill((candidate.personal ? Lab.violet : Lab.emerald).opacity(0.12))
            Circle().stroke(
                candidate.personal ? Lab.violet : Lab.emerald,
                lineWidth: active ? 3 : 1.5)
            Text(String(candidate.name.prefix(1)).uppercased())
                .font(.headline.bold().monospaced())
                .foregroundStyle(candidate.personal ? Lab.violet : Lab.emerald)
            if active {
                TimelineView(.animation(minimumInterval: 1 / 20)) { timeline in
                    let phase = timeline.date.timeIntervalSinceReferenceDate
                        .truncatingRemainder(dividingBy: 1.8) / 1.8
                    Circle()
                        .trim(from: 0.08, to: 0.42)
                        .stroke(
                            candidate.personal ? Lab.violet : Lab.cyan,
                            style: StrokeStyle(lineWidth: 3, lineCap: .round)
                        )
                        .rotationEffect(.degrees(phase * 360))
                        .padding(2)
                }
            }
        }
        .shadow(
            color: active ? (candidate.personal ? Lab.violet : Lab.emerald).opacity(0.5) : .clear,
            radius: 10)
        .accessibilityElement(children: .ignore)
        .accessibilityIdentifier(active ? "voice-lab-active-orb" : "voice-lab-orb-\(candidate.id)")
        .accessibilityLabel(active ? "Generating \(candidate.name)" : candidate.name)
        // The active treatment is painted inside this fixed frame. Never animate
        // scale or position: that made the currently generating voice appear to jump.
    }
}

private struct VoiceLabPulse: View {
    let active: Bool

    var body: some View {
        TimelineView(.animation(minimumInterval: active ? 1 / 30 : 1)) { timeline in
            let phase = timeline.date.timeIntervalSinceReferenceDate
            Canvas { context, size in
                let center = CGPoint(x: size.width / 2, y: size.height / 2)
                for ring in 0..<3 {
                    let wave = active ? (sin(phase * 2.5 + Double(ring)) + 1) / 2 : 0.25
                    let radius = 14 + CGFloat(ring) * 8 + CGFloat(wave) * 3
                    let rect = CGRect(
                        x: center.x - radius, y: center.y - radius,
                        width: radius * 2, height: radius * 2)
                    context.stroke(
                        Path(ellipseIn: rect),
                        with: .color(Lab.emerald.opacity(0.8 - Double(ring) * 0.2)),
                        lineWidth: ring == 0 ? 3 : 1.5)
                }
                let bolt = Text(Image(systemName: "waveform.path.ecg"))
                    .font(.system(size: 22, weight: .black))
                    .foregroundColor(Lab.emerald)
                context.draw(bolt, at: center)
            }
        }
        .accessibilityHidden(true)
    }
}
