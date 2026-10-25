import CoreAI
import Foundation

/// Loads a Core AI program and builds `T` from it, compiling it again once if Core AI cannot use its cached copy.
func loadProgram<T>(_ url: URL, on unit: ComputeUnitKind, _ build: (AIModel) throws -> T) async throws -> T {
    let options = SpecializationOptions(preferredComputeUnitKind: unit)
    do {
        return try build(try await AIModel(contentsOf: url, options: options))
    } catch where isStaleCompiledCopy(error) {
        // Core AI returns nil instead of recompiling when a cached program goes stale (seen 2026-10-25 after a restart and after other models compiled)
        guard try removeCompiledCopy(of: url) else { throw AnemoneError.staleCompileCache }
        do {
            return try build(try await AIModel(contentsOf: url, options: options))
        } catch where isStaleCompiledCopy(error) {
            throw AnemoneError.staleCompileCache
        }
    }
}

private func isStaleCompiledCopy(_ error: Error) -> Bool {
    "\(error)" == "nilError"
}

/// Deletes the program's entry, `coreai-cache/<macOS build>/<app>/<main.hash in hex>`; false if there is none.
private func removeCompiledCopy(of url: URL) throws -> Bool {
    let hash = try Data(contentsOf: url.appending(path: "main.hash")).map { String(format: "%02x", $0) }.joined()
    let files = FileManager.default
    let root = files.urls(for: .cachesDirectory, in: .userDomainMask)[0].appending(path: "coreai-cache")
    let entries = try files.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
        .map { $0.appending(path: ProcessInfo.processInfo.processName).appending(path: hash) }
        .filter { files.fileExists(atPath: $0.path) }
    for entry in entries { try files.removeItem(at: entry) }
    return !entries.isEmpty
}
