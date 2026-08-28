import Foundation

/// Minimal RFC4180-ish CSV support: no external dependency, handles quoted
/// fields (embedded commas/newlines/escaped quotes) on read and quotes
/// whatever needs it on write. Not a general-purpose CSV library -- just
/// enough to read manifest.csv and read/write hypotheses.csv correctly,
/// since `hypothesis` text can itself contain commas and quotes.
enum CSV {
    static func parse(_ content: String) -> [[String]] {
        var rows: [[String]] = []
        var field = ""
        var row: [String] = []
        var inQuotes = false
        let chars = Array(content)
        var i = 0
        while i < chars.count {
            let c = chars[i]
            if inQuotes {
                if c == "\"" {
                    if i + 1 < chars.count, chars[i + 1] == "\"" {
                        field.append("\"")
                        i += 1
                    } else {
                        inQuotes = false
                    }
                } else {
                    field.append(c)
                }
            } else if c == "\"" {
                inQuotes = true
            } else if c == "," {
                row.append(field)
                field = ""
            } else if c == "\n" {
                row.append(field)
                field = ""
                rows.append(row)
                row = []
            } else if c == "\r" {
                if i + 1 < chars.count, chars[i + 1] == "\n" {
                    // the following \n closes this record on the next iteration
                } else {
                    row.append(field)
                    field = ""
                    rows.append(row)
                    row = []
                }
            } else {
                field.append(c)
            }
            i += 1
        }
        if !field.isEmpty || !row.isEmpty {
            row.append(field)
            rows.append(row)
        }
        return rows
    }

    static func escapeField(_ value: String) -> String {
        if value.contains(",") || value.contains("\"") || value.contains("\n") || value.contains("\r") {
            return "\"" + value.replacingOccurrences(of: "\"", with: "\"\"") + "\""
        }
        return value
    }

    static func row(_ fields: [String]) -> String {
        fields.map(escapeField).joined(separator: ",")
    }
}
