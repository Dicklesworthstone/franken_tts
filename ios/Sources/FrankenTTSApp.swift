import SwiftUI

@main
struct FrankenTTSApp: App {
    init() {
        // Five team partitions, not the Mac's six: 2P+4E leaves one E core for UI and
        // audio (docs/IOS_APP_PLAN.md §1). Must be set before the first engine call.
        setenv("FTTS_INT8_THREADS", "5", 1)
    }

    var body: some Scene {
        WindowGroup {
            LabView()
                .preferredColorScheme(.dark)
        }
    }
}
