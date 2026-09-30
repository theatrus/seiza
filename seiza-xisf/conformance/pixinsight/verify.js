#include <pjsr/ColorSpace.jsh>
#include <pjsr/SampleType.jsh>
var BASE = "__BASE__";
var out = [];
function sampleK(x, y, c, seed, mk) { return (x * 7919 + y * 104729 + c * 1299709 + seed * 31) % (mk + 1); }
function show(value) {
   if (value === null || value === undefined) return String(value);
   try { if (typeof value.toArray == "function") return "[" + value.toArray().join(",") + "]"; } catch (e) {}
   try { if (value.length !== undefined && typeof value != "string") { var a = []; for (var i = 0; i < value.length; ++i) a.push(value.at ? value.at(i) : value[i]); return "[" + a.join(",") + "]"; } } catch (e) {}
   return String(value);
}
function openFile(path, hints) {
   var f = new FileFormatInstance(new FileFormat(".xisf", true, false));
   var d = f.open(path, hints);
   if (!d || d.length < 1) throw "open failed: " + path;
   return f;
}
function readPixels(path, hints) {
   var f = openFile(path, hints);
   var img = new Image(1, 1, 1, ColorSpace_Gray, 32, SampleType_Real);
   if (!f.readImage(img)) throw "readImage failed";
   f.close();
   return img;
}
function metadata(path) {
   var f = openFile(path, "");
   var props = {};
   var list = f.imageProperties;
   for (var i = 0; i < list.length; ++i) {
      var id = list[i][0];
      props[id] = list[i][1] + ":" + show(f.readImageProperty(id));
   }
   var unit = {};
   var ulist = f.properties;
   for (var i = 0; i < ulist.length; ++i) unit[ulist[i][0]] = show(f.readProperty(ulist[i][0]));
   var keys = f.keywords.map(function (k) { return k.name + "=" + k.value + "/" + k.comment; });
   var thumb = f.thumbnail;
   var result = { props: props, unit: unit, keys: keys, thumb: thumb ? thumb.width + "x" + thumb.height : "none",
                  cfa: show(f.colorFilterArray), rgbws: f.rgbws ? String(f.rgbws.gamma) : "none" };
   f.close();
   return result;
}
function compareMaps(label, a, b, skip) {
   var problems = [];
   for (var k in a) if (!skip(k) && a[k] !== b[k]) problems.push(label + " " + k + ": " + a[k] + " != " + b[k]);
   for (var k in b) if (!skip(k) && !(k in a)) problems.push(label + " extra " + k + "=" + b[k]);
   return problems;
}
var lines = File.readLines(BASE + "/to_pi/manifest.txt");
for (var n = 0; n < lines.length; ++n) {
   var line = lines[n];
   if (line.length == 0) continue;
   var f = line.split("|");
   var name = f[0], w = +f[1], h = +f[2], ch = +f[3], bits = +f[4], isFloat = f[5] == "1", seed = +f[6], source = f[7];
   var mk = isFloat ? 65535 : Math.pow(2, bits) - 1;
   var path = BASE + "/to_pi/" + name + ".xisf";
   try {
      var results = [];
      var pixelProblems = [];
      var hintsList = ["", "no-normalize"];
      for (var hi = 0; hi < hintsList.length; ++hi) {
         var img = readPixels(path, hintsList[hi]);
         var maxErr = 0;
         if (img.width != w || img.height != h || img.numberOfChannels != ch) throw "geometry " + img.width + "x" + img.height + "x" + img.numberOfChannels;
         for (var c = 0; c < ch; ++c) for (var y = 0; y < h; ++y) for (var x = 0; x < w; ++x) {
            var e = Math.abs(img.sample(x, y, c) - sampleK(x, y, c, seed, mk) / mk);
            if (e > maxErr) maxErr = e;
         }
         results.push("[" + (hintsList[hi] || "default") + "] max_error=" + maxErr.toExponential(3));
         if (maxErr > 1e-6) pixelProblems.push((hintsList[hi] || "default read") + " max_error=" + maxErr.toExponential(3));
      }
      var line2 = (pixelProblems.length ? "FAIL " : "ok   ") + name + " " + results.join(" ");
      if (source) {
         var a = metadata(BASE + "/from_pi/" + source + ".xisf");
         var b = metadata(path);
         var problems = pixelProblems.concat(compareMaps("prop", a.props, b.props, function (k) { return false; })
            .concat(compareMaps("unit", a.unit, b.unit, function (k) { return /^XISF:(Creat|Compression|Checksum|BlockAlign|OriginalCreation|MaxInlineBlockSize|OutputHints|ResourceURL)/.test(k); })));
         if (a.keys.join("\n") != b.keys.join("\n")) problems.push("keywords differ: " + a.keys.join(";") + " VS " + b.keys.join(";"));
         if (a.thumb != b.thumb) problems.push("thumbnail " + a.thumb + " != " + b.thumb);
         if (a.cfa != b.cfa) problems.push("cfa " + a.cfa + " != " + b.cfa);
         if (a.rgbws != b.rgbws) problems.push("rgbws " + a.rgbws + " != " + b.rgbws);
         line2 += " props=" + Object.keys(b.props).length + " keywords=" + b.keys.length + " thumb=" + b.thumb +
                  " OriginalCreationTime=" + (b.unit["XISF:OriginalCreationTime"] || "none") +
                  (problems.length ? "\n   PROBLEMS:\n   " + problems.join("\n   ") : " metadata=same");
      }
      else if (pixelProblems.length)
         line2 += "\n   PROBLEMS:\n   " + pixelProblems.join("\n   ");
      if (line2.indexOf("PROBLEMS") >= 0 && line2.indexOf("ok   ") == 0)
         line2 = "FAIL " + line2.substring(5);
      out.push(line2);
   } catch (e) {
      out.push("EXC " + name + ": " + e);
   }
}
File.writeTextFile(BASE + "/verify.txt", out.join("\n") + "\n");
