import Foundation

/// The functions of a compiled program that Core AI did not place on the ANE, read from its cache entry; nil if unreadable.
func functionsOffNeuralEngine(_ program: URL, functions: [String]) -> [String]? {
    // when the ANE compiler rejects a program Core AI runs it elsewhere without an error (lab 2026-10-27: a layer took 21 to 28 ms, not about 2)
    guard let entry = try? compiledCopies(of: program).first,
        let manifests = FileManager.default.enumerator(at: entry, includingPropertiesForKeys: nil)?
            .compactMap({ $0 as? URL }).filter({ $0.lastPathComponent == "manifest.plist" }),
        !manifests.isEmpty
    else { return nil }
    let regions = manifests.flatMap { manifest -> [String] in
        guard let data = try? Data(contentsOf: manifest),
            let plist = try? PropertyListSerialization.propertyList(from: data, format: nil)
        else { return [] }
        return strings(in: plist).filter { $0.contains("_ANE_region_") }
    }
    return functionsWithoutRegion(functions, regions: regions)
}

/// Functions with no ANE region: regions are named `<function>_<hash>_<index>_ANE_region_<n>_<m>`.
func functionsWithoutRegion(_ functions: [String], regions: [String]) -> [String] {
    functions.filter { name in !regions.contains { $0.hasPrefix(name + "_") } }
}

private func strings(in value: Any) -> [String] {
    switch value {
    case let text as String: [text]
    case let array as [Any]: array.flatMap(strings)
    case let dictionary as [String: Any]: dictionary.keys.map { $0 } + dictionary.values.flatMap(strings)
    default: []
    }
}
