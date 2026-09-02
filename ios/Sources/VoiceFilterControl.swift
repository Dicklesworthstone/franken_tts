import SwiftUI

struct VoiceFilterControl: View {
    @Binding var showsFavoritesOnly: Bool
    let allCount: Int
    let favoriteCount: Int

    var body: some View {
        HStack(spacing: 4) {
            segment(title: "All \(allCount)", selected: !showsFavoritesOnly) {
                showsFavoritesOnly = false
            }
            segment(title: "Favorites \(favoriteCount)", selected: showsFavoritesOnly) {
                showsFavoritesOnly = true
            }
        }
        .padding(4)
        .frame(maxWidth: 270)
        .background(Lab.panelSoft, in: Capsule())
        .overlay(Capsule().stroke(Lab.stroke, lineWidth: 1))
        .animation(.snappy(duration: 0.22), value: showsFavoritesOnly)
        .accessibilityElement(children: .contain)
    }

    private func segment(
        title: String,
        selected: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: Lab.typeSize(10), weight: .bold, design: .rounded))
                .lineLimit(1)
                .frame(maxWidth: .infinity)
                .padding(.horizontal, 10)
                .frame(height: 36)
                .foregroundStyle(selected ? Color.white : Lab.textSecondary)
                .background(selected ? Lab.emeraldDeep : Color.clear, in: Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}
