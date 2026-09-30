#include <pjsr/ColorSpace.jsh>
#include <pjsr/UndoFlag.jsh>
#include <pjsr/PropertyType.jsh>
#include <pjsr/PropertyAttribute.jsh>
var OUT = "__BASE__/from_pi";
var log = [];
var manifest = [];
function maxk(bits, isFloat) { return isFloat ? 65535 : Math.pow(2, bits) - 1; }
function sampleK(x, y, c, seed, mk) { return (x * 7919 + y * 104729 + c * 1299709 + seed * 31) % (mk + 1); }
function makeWindow(w, h, ch, bits, isFloat, seed, id) {
   var win = new ImageWindow(w, h, ch, bits, isFloat, ch == 3, id);
   var view = win.mainView;
   view.beginProcess(UndoFlag_NoSwapFile);
   var img = view.image;
   var mk = maxk(bits, isFloat);
   for (var c = 0; c < ch; ++c)
      for (var y = 0; y < h; ++y)
         for (var x = 0; x < w; ++x)
            img.setSample(sampleK(x, y, c, seed, mk) / mk, x, y, c);
   view.endProcess();
   win.__seed = seed;
   return win;
}
function save(name, win, hints, extra) {
   var path = OUT + "/" + name + ".xisf";
   try {
      var ok = win.saveAs(path, false, false, false, false, hints);
      var img = win.mainView.image;
      manifest.push([name, img.width, img.height, img.numberOfChannels, img.bitsPerSample, img.isReal ? 1 : 0, win.__seed, hints, extra || ""].join("|"));
      log.push((ok ? "ok   " : "FAIL ") + name + " [" + hints + "]");
   } catch (e) {
      log.push("EXC  " + name + ": " + e);
   }
}
function run() {
   var formats = [[8, false, "u8"], [16, false, "u16"], [32, false, "u32"], [32, true, "f32"], [64, true, "f64"]];
   var seed = 1;
   for (var i = 0; i < formats.length; ++i)
      for (var ch = 1; ch <= 3; ch += 2) {
         var f = formats[i];
         var win = makeWindow(23, 17, ch, f[0], f[1], seed, "base_" + f[2] + "_" + ch);
         save("base_" + f[2] + "_" + ch, win, "no-compression checksums sha1");
         win.forceClose();
         ++seed;
      }
   var codecs = ["zlib", "zlib+sh", "lz4", "lz4+sh", "lz4hc", "lz4hc+sh", "zstd", "zstd+sh"];
   var sums = ["sha1", "sha256", "sha512"];
   for (var i = 0; i < codecs.length; ++i) {
      var spec = [[16, false, "u16", 3], [32, true, "f32", 1], [64, true, "f64", 3]];
      for (var j = 0; j < spec.length; ++j) {
         var s = spec[j];
         var name = "codec_" + codecs[i].replace("+", "sh_") + "_" + s[2] + "_" + s[3];
         var win = makeWindow(61, 47, s[3], s[0], s[1], seed, "w" + seed);
         save(name, win, "compression-codec " + codecs[i] + " compression-level " + (10 + 12 * i) + " checksums " + sums[(i + j) % 3]);
         win.forceClose();
         ++seed;
      }
   }
   var win = makeWindow(8, 6, 3, 8, false, seed, "embedded"); ++seed;
   save("embedded_u8_3", win, "embedded-data no-compression checksums sha256");
   save("embedded_zstd_u8_3", win, "embedded-data compression-codec zstd+sh checksums sha1");
   save("unaligned_u8_3", win, "no-block-alignment no-compression no-checksums");
   save("align16_u8_3", win, "block-alignment 16 compression-codec lz4 no-checksums");
   win.forceClose();

   var win = makeWindow(40, 30, 1, 16, false, seed, "props"); ++seed;
   var v = win.mainView;
   var A = PropertyAttribute_Storable | PropertyAttribute_Permanent;
   function prop(id, value, type) {
      try { v.setPropertyValue(id, value, type, A); } catch (e) { log.push("EXC prop " + id + ": " + e); }
   }
   prop("Test:Int8", -12, PropertyType_Int8);
   prop("Test:UInt8", 250, PropertyType_UInt8);
   prop("Test:Int16", -30000, PropertyType_Int16);
   prop("Test:UInt16", 65000, PropertyType_UInt16);
   prop("Test:Int32", -2000000000, PropertyType_Int32);
   prop("Test:UInt32", 4000000000, PropertyType_UInt32);
   prop("Test:Int64", -9007199254740991, PropertyType_Int64);
   prop("Test:UInt64", 9007199254740991, PropertyType_UInt64);
   prop("Test:Float32", 1.25, PropertyType_Float32);
   prop("Test:Float64", -3.141592653589793, PropertyType_Float64);
   prop("Test:Boolean", true, PropertyType_Boolean);
   prop("Test:String", "M 42 & <Orion> \"nebula\" é—", PropertyType_String);
   prop("Test:TimePoint", "2026-09-29T21:30:15.25Z", PropertyType_TimePoint);
   var small = new Vector(3); small.at(0, 1.5); small.at(1, -2.25); small.at(2, 1e-300);
   prop("Test:SmallVector", small, PropertyType_F64Vector);
   var big = new Vector(5000); for (var k = 0; k < 5000; ++k) big.at(k, k * 0.5 - 1000);
   prop("Test:BigVector", big, PropertyType_F64Vector);
   var m = new Matrix(3, 4); for (var r = 0; r < 3; ++r) for (var c = 0; c < 4; ++c) m.at(r, c, r * 10 + c);
   prop("Test:Matrix", m, PropertyType_F64Matrix);
   var iv = new Vector(4); iv.at(0, -1); iv.at(1, 0); iv.at(2, 7); iv.at(3, 123456);
   prop("Test:I32Vector", iv, PropertyType_I32Vector);
   prop("Test:ByteArray", new ByteArray("hello world"), PropertyType_ByteArray);
   prop("Observation:Object:Name", "M 42", PropertyType_String);
   prop("Observation:Center:RA", 83.8221, PropertyType_Float64);
   prop("Observation:Center:Dec", -5.3911, PropertyType_Float64);
   prop("Observation:Location:Longitude", -122.4, PropertyType_Float64);
   win.keywords = [
      new FITSKeyword("OBJECT", "'M 42'", "Target"),
      new FITSKeyword("EXPTIME", "300.", "seconds"),
      new FITSKeyword("GAIN", "100", ""),
      new FITSKeyword("QUOTED", "'it''s'", "quote test"),
      new FITSKeyword("HISTORY", "", "first history"),
      new FITSKeyword("HISTORY", "", "second history"),
      new FITSKeyword("BAYERPAT", "'RGGB'", "")
   ];
   save("props_u16_1", win, "compression-codec zstd+sh checksums sha1 max-inline-block-size 64");
   save("props_plain_u16_1", win, "no-compression no-checksums");
   win.forceClose();

   try {
      var win = makeWindow(16, 12, 3, 32, true, seed, "lab"); ++seed;
      var view = win.mainView;
      view.beginProcess(UndoFlag_NoSwapFile);
      view.image.colorSpace = ColorSpace_CIELab;
      log.push("lab colorSpace now " + view.image.colorSpace);
      view.endProcess();
      save("lab_f32_3", win, "no-compression", "lab");
      win.forceClose();
   } catch (e) { log.push("EXC lab: " + e); }

   try {
      var fmt = new FileFormat(".xisf", false, true);
      var f = new FileFormatInstance(fmt);
      if (!f.create(OUT + "/multi_u16_1.xisf", "image-ids first,second compression-codec zlib+sh checksums sha512"))
         throw "create failed";
      var d = new ImageDescription; d.bitsPerSample = 16; d.ieeefpSampleFormat = false;
      var w1 = makeWindow(10, 9, 1, 16, false, seed, "m1"); f.setOptions(d); f.writeImage(w1.mainView.image);
      manifest.push(["multi_u16_1#0", 10, 9, 1, 16, 0, seed, "multi", ""].join("|")); ++seed;
      var w2 = makeWindow(12, 5, 1, 16, false, seed, "m2"); f.setOptions(d); f.writeImage(w2.mainView.image);
      manifest.push(["multi_u16_1#1", 12, 5, 1, 16, 0, seed, "multi", ""].join("|")); ++seed;
      f.close(); w1.forceClose(); w2.forceClose();
      log.push("ok   multi");
   } catch (e) { log.push("EXC multi: " + e); }
}
try { run(); } catch (e) { log.push("EXC run: " + e); }
File.writeTextFile(OUT + "/manifest.txt", manifest.join("\n") + "\n");
File.writeTextFile(OUT + "/log.txt", log.join("\n") + "\n");
