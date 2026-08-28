import Foundation

struct ManifestRow {
    let uttID: String
    let wavPath: String
}

enum ManifestError: Error, CustomStringConvertible {
    case fileNotFound(String)
    case empty(String)
    case missingColumn(String, String)

    var description: String {
        switch self {
        case .fileNotFound(let path):
            return "manifest not found at \(path)"
        case .empty(let path):
            return "manifest at \(path) has no rows"
        case .missingColumn(let column, let path):
            return "manifest at \(path) has no '\(column)' column"
        }
    }
}

enum Manifest {
    /// Reads utt_id/wav_path out of a manifest CSV by header name, not
    /// column position -- utt_id is the join key everywhere in this
    /// project, and other manifest columns are free to move or grow
    /// without breaking this reader.
    static func read(path: String) throws -> [ManifestRow] {
        guard FileManager.default.fileExists(atPath: path) else {
            throw ManifestError.fileNotFound(path)
        }
        let content = try String(contentsOfFile: path, encoding: .utf8)
        let table = CSV.parse(content).filter { !($0.count == 1 && $0[0].isEmpty) }
        guard let header = table.first else {
            throw ManifestError.empty(path)
        }
        guard let uttIDIndex = header.firstIndex(of: "utt_id") else {
            throw ManifestError.missingColumn("utt_id", path)
        }
        guard let wavPathIndex = header.firstIndex(of: "wav_path") else {
            throw ManifestError.missingColumn("wav_path", path)
        }
        var rows: [ManifestRow] = []
        for record in table.dropFirst() {
            guard record.count > max(uttIDIndex, wavPathIndex) else { continue }
            rows.append(ManifestRow(uttID: record[uttIDIndex], wavPath: record[wavPathIndex]))
        }
        return rows
    }

    /// utt_ids already present in an existing hypotheses.csv -- the
    /// checkpoint/resume mechanism. A missing file just means nothing has
    /// run yet, not an error.
    static func completedUttIDs(hypothesesPath: String) throws -> Set<String> {
        guard FileManager.default.fileExists(atPath: hypothesesPath) else {
            return []
        }
        let content = try String(contentsOfFile: hypothesesPath, encoding: .utf8)
        let table = CSV.parse(content).filter { !($0.count == 1 && $0[0].isEmpty) }
        guard let header = table.first, let uttIDIndex = header.firstIndex(of: "utt_id") else {
            return []
        }
        var ids = Set<String>()
        for record in table.dropFirst() where record.count > uttIDIndex {
            ids.insert(record[uttIDIndex])
        }
        return ids
    }
}
