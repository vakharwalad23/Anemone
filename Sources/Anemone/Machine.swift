import Foundation
import IOKit.ps

/// The Mac a bench ran on, saved with every result so numbers from different Macs can be compared.
package struct Machine: Codable {
    package let chip: String
    package let memoryGB: Int
    package let macOS: String
    package let build: String
    package let power: String

    package static func current() -> Machine {
        let version = ProcessInfo.processInfo.operatingSystemVersion
        return Machine(
            chip: sysctlString("machdep.cpu.brand_string"), memoryGB: Int(ProcessInfo.processInfo.physicalMemory >> 30),
            macOS: "\(version.majorVersion).\(version.minorVersion).\(version.patchVersion)", build: sysctlString("kern.osversion"),
            power: powerSource())
    }
}

func sysctlString(_ name: String) -> String {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0 else { return "" }
    var buffer = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return "" }
    return String(utf8String: buffer) ?? ""
}

func powerSource() -> String {
    let info = IOPSCopyPowerSourcesInfo().takeRetainedValue()
    let source = IOPSGetProvidingPowerSourceType(info).takeUnretainedValue() as String
    return source == kIOPSACPowerValue ? "AC" : "battery"
}
