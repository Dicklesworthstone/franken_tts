import SwiftUI
import Foundation

@main
struct FrankenTTSApp: App {
    init() {
        // Five team partitions, not the Mac's six: 2P+4E leaves one E core for UI and
        // audio (docs/IOS_APP_PLAN.md §1). A launch override is preserved so physical-device
        // A/B runs measure distinct team sizes instead of silently testing five versus five.
        if ProcessInfo.processInfo.environment["FTTS_INT8_THREADS"] == nil {
            setenv("FTTS_INT8_THREADS", "5", 1)
        }
    }

    var body: some Scene {
        WindowGroup {
            LabView()
                .preferredColorScheme(.dark)
        }
    }
}
