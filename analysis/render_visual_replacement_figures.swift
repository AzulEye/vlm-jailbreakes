#!/usr/bin/env swift
import AppKit
import CoreGraphics
import Foundation
import ImageIO

// Renders one PDF "figure" per row from analysis/datasets/visual_replacement_top_behaviors_*.csv.
// Intended to match the layout style used in the paper appendix figures.

struct JudgeEntry {
    let judgeModel: String
    let classification: Int?
    let safetyScore: Double?
    let rationale: String
}

struct AttackRow {
    let attackId: String
    let behaviorId: String
    let resultsDir: String
    let run: Int
    let behavior: String
    let attackPrompt: String
    let imagesBySlot: [String: [String]]
    let slotValues: [String: String]
    let slotReplacements: [String: String]
    let judgeEntries: [JudgeEntry]
    let judgeAvgSafety: Double?
}

struct CLIOptions {
    let csvPath: String
    let outDir: String
    let limit: Int?
    let overwrite: Bool
    let responseChars: Int
}

func parseArgs() -> CLIOptions {
    var csvPath = "analysis/datasets/visual_replacement_top_behaviors_qwen3-vl-32b-instruct.csv"
    var outDir = "figures/appendix/visual_replacement_top_behaviors"
    var limit: Int? = nil
    var overwrite = false
    var responseChars = 700

    var i = 1
    let args = CommandLine.arguments
    while i < args.count {
        let a = args[i]
        func needValue() -> String {
            guard i + 1 < args.count else {
                fputs("Missing value for \(a)\n", stderr)
                exit(2)
            }
            i += 1
            return args[i]
        }
        switch a {
        case "--csv":
            csvPath = needValue()
        case "--out-dir":
            outDir = needValue()
        case "--limit":
            limit = Int(needValue())
        case "--response-chars":
            responseChars = Int(needValue()) ?? responseChars
        case "--overwrite":
            overwrite = true
        case "--help", "-h":
            print(
                """
                Usage:
                  swift analysis/render_visual_replacement_figures.swift \\
                    --csv analysis/datasets/visual_replacement_top_behaviors_qwen3-vl-32b-instruct.csv \\
                    --out-dir figures/appendix/visual_replacement_top_behaviors \\
                    --limit 5 \\
                    --overwrite \\
                    --response-chars 700
                """
            )
            exit(0)
        default:
            break
        }
        i += 1
    }

    return CLIOptions(
        csvPath: csvPath,
        outDir: outDir,
        limit: limit,
        overwrite: overwrite,
        responseChars: responseChars
    )
}

// RFC4180-ish CSV parser that supports quoted fields with commas and newlines.
func parseCSV(_ text: String) -> [[String]] {
    var rows: [[String]] = []
    var row: [String] = []
    var field = ""
    var inQuotes = false

    var i = text.startIndex
    while i < text.endIndex {
        let ch = text[i]
        if ch == "\"" {
            if inQuotes {
                let next = text.index(after: i)
                if next < text.endIndex && text[next] == "\"" {
                    field.append("\"")
                    i = next
                } else {
                    inQuotes = false
                }
            } else {
                inQuotes = true
            }
        } else if ch == "," && !inQuotes {
            row.append(field)
            field = ""
        } else if (ch == "\n" || ch == "\r") && !inQuotes {
            row.append(field)
            field = ""
            rows.append(row)
            row = []
            if ch == "\r" {
                let next = text.index(after: i)
                if next < text.endIndex && text[next] == "\n" {
                    i = next
                }
            }
        } else {
            field.append(ch)
        }
        i = text.index(after: i)
    }
    row.append(field)
    rows.append(row)

    // Drop trailing empty row if present.
    while let last = rows.last, last.allSatisfy({ $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) {
        rows.removeLast()
    }
    return rows
}

func safeFilename(_ s: String) -> String {
    let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return s.unicodeScalars.map { allowed.contains($0) ? Character($0) : "_" }.reduce("") { $0 + String($1) }
}

func jsonObject(_ s: String) -> Any? {
    let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return nil }
    guard let data = trimmed.data(using: .utf8) else { return nil }
    return try? JSONSerialization.jsonObject(with: data, options: [])
}

func jsonDictStringString(_ s: String) -> [String: String] {
    guard let obj = jsonObject(s) as? [String: Any] else { return [:] }
    var out: [String: String] = [:]
    for (k, v) in obj {
        if let str = v as? String { out[k] = str }
    }
    return out
}

func jsonDictStringStringArray(_ s: String) -> [String: [String]] {
    guard let obj = jsonObject(s) as? [String: Any] else { return [:] }
    var out: [String: [String]] = [:]
    for (k, v) in obj {
        if let arr = v as? [Any] {
            out[k] = arr.compactMap { $0 as? String }
        }
    }
    return out
}

func jsonJudgeEntries(_ s: String) -> [JudgeEntry] {
    guard let arr = jsonObject(s) as? [Any] else { return [] }
    var out: [JudgeEntry] = []
    for item in arr {
        guard let d = item as? [String: Any] else { continue }
        let model = (d["judge_model"] as? String) ?? ""
        let cls = d["classification"] as? Int
        let score: Double?
        if let v = d["safety_score"] as? Double { score = v }
        else if let v = d["safety_score"] as? Int { score = Double(v) }
        else if let v = d["safety_score"] as? String { score = Double(v) }
        else { score = nil }
        let rationale = (d["rationale"] as? String) ?? ""
        out.append(JudgeEntry(judgeModel: model, classification: cls, safetyScore: score, rationale: rationale))
    }
    return out
}

func toInt(_ s: String) -> Int? { Int(s.trimmingCharacters(in: .whitespacesAndNewlines)) }
func toDouble(_ s: String) -> Double? { Double(s.trimmingCharacters(in: .whitespacesAndNewlines)) }

func loadCGImage(path: String) -> CGImage? {
    let url = URL(fileURLWithPath: path)
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(src, 0, nil)
}

func aspectFitRect(contentSize: CGSize, in rect: CGRect) -> CGRect {
    if contentSize.width <= 0 || contentSize.height <= 0 {
        return rect
    }
    let scale = min(rect.width / contentSize.width, rect.height / contentSize.height)
    let w = contentSize.width * scale
    let h = contentSize.height * scale
    let x = rect.origin.x + (rect.width - w) / 2
    let y = rect.origin.y + (rect.height - h) / 2
    return CGRect(x: x, y: y, width: w, height: h)
}

func classLabel(_ cls: Int?) -> String {
    switch cls {
    case 3: return "Full Compliance"
    case 2: return "Partial Compliance"
    case 1: return "Misunderstanding"
    case 0: return "Refusal"
    default: return "Unknown"
    }
}

func classColor(_ cls: Int?) -> NSColor {
    switch cls {
    case 3: return NSColor(calibratedRed: 0.84, green: 0.18, blue: 0.17, alpha: 1.0)
    case 2: return NSColor(calibratedRed: 0.96, green: 0.62, blue: 0.04, alpha: 1.0)
    case 1: return NSColor(calibratedRed: 0.45, green: 0.55, blue: 0.64, alpha: 1.0)
    case 0: return NSColor(calibratedRed: 0.45, green: 0.55, blue: 0.64, alpha: 1.0)
    default: return NSColor(calibratedWhite: 0.5, alpha: 1.0)
    }
}

func truncate(_ s: String, maxChars: Int) -> String {
    let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.count <= maxChars { return trimmed }
    let idx = trimmed.index(trimmed.startIndex, offsetBy: maxChars)
    return String(trimmed[..<idx]) + "\n[... truncated for safety ...]"
}

func drawWrappedText(_ text: String, in rect: CGRect, font: NSFont, color: NSColor, alignment: NSTextAlignment = .left) {
    let p = NSMutableParagraphStyle()
    p.alignment = alignment
    p.lineBreakMode = .byWordWrapping
    let attrs: [NSAttributedString.Key: Any] = [
        .font: font,
        .foregroundColor: color,
        .paragraphStyle: p,
    ]
    (text as NSString).draw(with: rect, options: [.usesLineFragmentOrigin, .usesFontLeading], attributes: attrs)
}

func drawCenteredText(_ text: String, in rect: CGRect, font: NSFont, color: NSColor) {
    drawWrappedText(text, in: rect, font: font, color: color, alignment: .center)
}

func drawRoundedPill(_ text: String, at origin: CGPoint, height: CGFloat, paddingX: CGFloat, color: NSColor, font: NSFont) -> CGRect {
    let attrs: [NSAttributedString.Key: Any] = [.font: font]
    let textSize = (text as NSString).size(withAttributes: attrs)
    let w = textSize.width + 2 * paddingX
    let rect = CGRect(x: origin.x, y: origin.y, width: w, height: height)

    color.setFill()
    let path = NSBezierPath(roundedRect: rect, xRadius: height / 2, yRadius: height / 2)
    path.fill()

    let textRect = CGRect(x: rect.origin.x, y: rect.origin.y + (height - textSize.height) / 2 - 0.5, width: rect.width, height: height)
    drawCenteredText(text, in: textRect, font: font, color: .white)
    return rect
}

func readVLMReply(resultsDir: String, run: Int) -> String {
    let url = URL(fileURLWithPath: resultsDir).appendingPathComponent("vlm_reply_qwen3-vl-32b-instruct.json")
    guard let data = try? Data(contentsOf: url) else { return "" }
    guard let obj = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] else { return "" }
    guard let runs = obj["runs"] as? [Any] else { return "" }
    for item in runs {
        guard let d = item as? [String: Any] else { continue }
        if (d["run"] as? Int) == run {
            return (d["reply"] as? String) ?? ""
        }
    }
    return ""
}

func renderFigure(row: AttackRow, outURL: URL, responseChars: Int) throws {
    let pageWidth: CGFloat = 928
    let pageHeight: CGFloat = 1129
    var mediaBox = CGRect(x: 0, y: 0, width: pageWidth, height: pageHeight)

    guard let consumer = CGDataConsumer(url: outURL as CFURL) else {
        throw NSError(domain: "render", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create CGDataConsumer for \(outURL.path)"])
    }
    guard let ctx = CGContext(consumer: consumer, mediaBox: &mediaBox, nil) else {
        throw NSError(domain: "render", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create PDF CGContext"])
    }

    ctx.beginPDFPage(nil)
    let nsctx = NSGraphicsContext(cgContext: ctx, flipped: false)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = nsctx

    // Background
    NSColor.white.setFill()
    NSBezierPath(rect: mediaBox).fill()

    let marginX: CGFloat = 42
    let marginTop: CGFloat = 32
    let marginBottom: CGFloat = 32
    let contentWidth = pageWidth - 2 * marginX

    var yTop: CGFloat = pageHeight - marginTop

    let titleFont = NSFont.boldSystemFont(ofSize: 18)
    let headerFont = NSFont.boldSystemFont(ofSize: 12.5)
    let bodyFont = NSFont.systemFont(ofSize: 9.5)
    let smallFont = NSFont.systemFont(ofSize: 8.8)
    let monoFont = NSFont.monospacedSystemFont(ofSize: 8.0, weight: .regular)

    func advance(_ h: CGFloat) { yTop -= h }

    // Title
    let title = "Visual Object Replacement Attack Example: qwen3-vl-32b-instruct"
    let titleRect = CGRect(x: marginX, y: yTop - 26, width: contentWidth, height: 26)
    drawCenteredText(title, in: titleRect, font: titleFont, color: .black)
    advance(34)

    // Subtitle (behavior id)
    let subRect = CGRect(x: marginX, y: yTop - 16, width: contentWidth, height: 16)
    drawCenteredText(row.behaviorId.replacingOccurrences(of: "_", with: " "), in: subRect, font: NSFont.systemFont(ofSize: 11), color: NSColor(calibratedWhite: 0.25, alpha: 1.0))
    advance(28)

    // Section: images
    let imgHeaderRect = CGRect(x: marginX, y: yTop - 16, width: contentWidth, height: 16)
    drawCenteredText("Contextual Images with Replaced Objects", in: imgHeaderRect, font: headerFont, color: .black)
    advance(22)

    let slotKeys = row.imagesBySlot.keys.sorted()
    let slotCount = max(1, slotKeys.count)
    let imagesAreaHeight: CGFloat = 420
    let rowHeight = imagesAreaHeight / CGFloat(slotCount)
    let labelWidth: CGFloat = 155
    let colGap: CGFloat = 14
    let imgCols = 3
    let imagesWidth = contentWidth - labelWidth - 10
    let cellWidth = (imagesWidth - CGFloat(imgCols - 1) * colGap) / CGFloat(imgCols)
    let cellHeight = rowHeight - 8

    for (idx, slot) in slotKeys.enumerated() {
        let rowTop = yTop - CGFloat(idx) * rowHeight
        let labelRect = CGRect(x: marginX, y: rowTop - cellHeight, width: labelWidth, height: cellHeight)

        let orig = row.slotValues[slot] ?? ""
        let repl = row.slotReplacements[slot] ?? ""
        let label = orig.isEmpty ? "\(slot)" : "\(slot): \"\(orig)\"  \(repl)"
        drawWrappedText(label, in: labelRect.insetBy(dx: 0, dy: 20), font: bodyFont, color: .black, alignment: .left)

        let paths = row.imagesBySlot[slot] ?? []
        for c in 0..<imgCols {
            let x = marginX + labelWidth + 10 + CGFloat(c) * (cellWidth + colGap)
            let rect = CGRect(x: x, y: rowTop - cellHeight, width: cellWidth, height: cellHeight)
            NSColor(calibratedWhite: 0.96, alpha: 1.0).setFill()
            NSBezierPath(rect: rect).fill()

            if c < paths.count, let img = loadCGImage(path: paths[c]) {
                let fit = aspectFitRect(contentSize: CGSize(width: img.width, height: img.height), in: rect.insetBy(dx: 3, dy: 3))
                ctx.draw(img, in: fit)
            }

            NSColor(calibratedWhite: 0.75, alpha: 1.0).setStroke()
            NSBezierPath(rect: rect).stroke()
        }
    }
    advance(imagesAreaHeight + 14)

    // Section: attack prompt
    let promptHeaderRect = CGRect(x: marginX, y: yTop - 16, width: contentWidth, height: 16)
    drawCenteredText("Attack Prompt (Text Given to VLM)", in: promptHeaderRect, font: headerFont, color: .black)
    advance(22)

    let promptBoxH: CGFloat = 160
    let promptBoxRect = CGRect(x: marginX, y: yTop - promptBoxH, width: contentWidth, height: promptBoxH)
    NSColor(calibratedWhite: 0.98, alpha: 1.0).setFill()
    NSBezierPath(rect: promptBoxRect).fill()
    NSColor(calibratedRed: 0.12, green: 0.40, blue: 0.85, alpha: 1.0).setStroke()
    NSBezierPath(rect: promptBoxRect).stroke()

    ctx.saveGState()
    ctx.clip(to: promptBoxRect.insetBy(dx: 6, dy: 6))
    drawWrappedText(truncate(row.attackPrompt, maxChars: 1400), in: promptBoxRect.insetBy(dx: 8, dy: 8), font: monoFont, color: .black, alignment: .left)
    ctx.restoreGState()
    advance(promptBoxH + 14)

    // Section: decoded message (use the original behavior text)
    let decodedHeaderRect = CGRect(x: marginX, y: yTop - 16, width: contentWidth, height: 16)
    drawCenteredText("Decoded Hidden Message", in: decodedHeaderRect, font: headerFont, color: .black)
    advance(22)

    let decodedH: CGFloat = 55
    let decodedRect = CGRect(x: marginX + 60, y: yTop - decodedH, width: contentWidth - 120, height: decodedH)
    NSColor.white.setFill()
    NSBezierPath(rect: decodedRect).fill()
    NSColor(calibratedRed: 0.96, green: 0.52, blue: 0.08, alpha: 1.0).setStroke()
    NSBezierPath(roundedRect: decodedRect, xRadius: 8, yRadius: 8).stroke()
    drawCenteredText("\"\(row.behavior)\"", in: decodedRect.insetBy(dx: 10, dy: 8), font: bodyFont, color: .black)
    advance(decodedH + 14)

    // Section: response
    let respHeaderRect = CGRect(x: marginX, y: yTop - 16, width: contentWidth, height: 16)
    drawCenteredText("VLM Response (truncated for safety)", in: respHeaderRect, font: headerFont, color: .black)
    advance(22)

    let responseBoxH: CGFloat = 130
    let responseRect = CGRect(x: marginX + 30, y: yTop - responseBoxH, width: contentWidth - 60, height: responseBoxH)
    NSColor(calibratedWhite: 0.97, alpha: 1.0).setFill()
    NSBezierPath(rect: responseRect).fill()
    NSColor(calibratedWhite: 0.65, alpha: 1.0).setStroke()
    NSBezierPath(roundedRect: responseRect, xRadius: 8, yRadius: 8).stroke()

    let rawReply = readVLMReply(resultsDir: row.resultsDir, run: row.run)
    let reply = rawReply.isEmpty ? "[missing reply]" : truncate(rawReply, maxChars: responseChars)
    ctx.saveGState()
    ctx.clip(to: responseRect.insetBy(dx: 8, dy: 8))
    drawWrappedText(reply, in: responseRect.insetBy(dx: 10, dy: 10), font: smallFont, color: .black, alignment: .left)
    ctx.restoreGState()
    advance(responseBoxH + 14)

    // Section: judges
    let judgeHeaderRect = CGRect(x: marginX, y: yTop - 16, width: contentWidth, height: 16)
    drawCenteredText("Safety Judge Evaluations", in: judgeHeaderRect, font: headerFont, color: .black)
    advance(22)

    let judgeAreaH: CGFloat = max(110, pageHeight - marginBottom - (pageHeight - marginTop - yTop) - marginBottom)
    let judgeRect = CGRect(x: marginX, y: yTop - judgeAreaH, width: contentWidth, height: judgeAreaH)
    // no bounding box; draw entries directly

    var jYTop = yTop
    let entryH: CGFloat = 32
    for je in row.judgeEntries.prefix(3) {
        let lineRect = CGRect(x: marginX, y: jYTop - entryH, width: contentWidth, height: entryH)
        let modelName = je.judgeModel.isEmpty ? "judge" : je.judgeModel

        // Left: model name
        drawWrappedText(modelName + ":", in: CGRect(x: lineRect.minX, y: lineRect.minY + 10, width: 260, height: 14), font: bodyFont, color: .black, alignment: .left)

        // Pill + score
        let pillText: String
        if let sc = je.safetyScore {
            pillText = "\(classLabel(je.classification))  (s=\(String(format: \"%.1f\", sc)))"
        } else {
            pillText = classLabel(je.classification)
        }
        _ = drawRoundedPill(
            pillText,
            at: CGPoint(x: lineRect.minX + 265, y: lineRect.minY + 6),
            height: 18,
            paddingX: 10,
            color: classColor(je.classification),
            font: NSFont.boldSystemFont(ofSize: 8.8)
        )

        // Rationale
        let rat = je.rationale.trimmingCharacters(in: .whitespacesAndNewlines)
        if !rat.isEmpty {
            drawWrappedText(
                "Rationale: \(rat)",
                in: CGRect(x: lineRect.minX, y: lineRect.minY - 8, width: contentWidth, height: 20),
                font: smallFont,
                color: NSColor(calibratedWhite: 0.25, alpha: 1.0),
                alignment: .left
            )
        }

        jYTop -= (entryH + 12)
    }

    // Footer: avg safety score (optional)
    if let avg = row.judgeAvgSafety {
        let footerRect = CGRect(x: marginX, y: marginBottom - 6, width: contentWidth, height: 14)
        drawCenteredText("Avg judge safety score (this run): \(String(format: \"%.2f\", avg))", in: footerRect, font: smallFont, color: NSColor(calibratedWhite: 0.35, alpha: 1.0))
    }

    NSGraphicsContext.restoreGraphicsState()
    ctx.endPDFPage()
    ctx.closePDF()
}

func loadRows(csvPath: String) throws -> [AttackRow] {
    let url = URL(fileURLWithPath: csvPath)
    let text = try String(contentsOf: url, encoding: .utf8)
    let rows = parseCSV(text)
    guard rows.count >= 2 else { return [] }

    let header = rows[0]
    var index: [String: Int] = [:]
    for (i, h) in header.enumerated() { index[h] = i }
    func get(_ row: [String], _ key: String) -> String {
        guard let i = index[key], i < row.count else { return "" }
        return row[i]
    }

    var out: [AttackRow] = []
    for r in rows.dropFirst() {
        let run = toInt(get(r, "run")) ?? 0
        let judgeAvg = toDouble(get(r, "judge_avg_safety_score"))
        let row = AttackRow(
            attackId: get(r, "attack_id"),
            behaviorId: get(r, "behavior_id"),
            resultsDir: get(r, "results_dir"),
            run: run,
            behavior: get(r, "behavior"),
            attackPrompt: get(r, "attack_prompt"),
            imagesBySlot: jsonDictStringStringArray(get(r, "images_by_slot_local_json")),
            slotValues: jsonDictStringString(get(r, "slot_values_json")),
            slotReplacements: jsonDictStringString(get(r, "slot_replacements_json")),
            judgeEntries: jsonJudgeEntries(get(r, "judge_per_model_json")),
            judgeAvgSafety: judgeAvg
        )
        out.append(row)
    }
    return out
}

let opts = parseArgs()
let fm = FileManager.default
let outDirURL = URL(fileURLWithPath: opts.outDir)
try fm.createDirectory(at: outDirURL, withIntermediateDirectories: true)

let rows = try loadRows(csvPath: opts.csvPath)
let limited = opts.limit != nil ? Array(rows.prefix(opts.limit!)) : rows
var rendered = 0

for row in limited {
    autoreleasepool {
        let fname = "visual_replacement_qwen3-vl-32b-instruct_\(safeFilename(row.behaviorId)).pdf"
        let outURL = outDirURL.appendingPathComponent(fname)
        if !opts.overwrite && fm.fileExists(atPath: outURL.path) {
            return
        }
        do {
            try renderFigure(row: row, outURL: outURL, responseChars: opts.responseChars)
            rendered += 1
        } catch {
            fputs("Failed rendering \(row.behaviorId): \(error)\n", stderr)
        }
    }
}

print("Rendered \(rendered) PDF(s) to \(outDirURL.path)")

