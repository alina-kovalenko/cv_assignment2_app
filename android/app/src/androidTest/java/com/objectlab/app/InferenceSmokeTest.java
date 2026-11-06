package com.objectlab.app;

import android.app.Instrumentation;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.test.InstrumentationTestCase;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Field;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Locale;

import org.json.JSONArray;
import org.json.JSONObject;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OnnxValue;

/** On-device contract and UI smoke tests. Run on a device/emulator with generated ONNX assets. */
public final class InferenceSmokeTest extends InstrumentationTestCase {
    private static final String[] MODELS = {"baseline.onnx", "sota.onnx"};
    private Context app;
    private OrtEnvironment environment;

    @Override protected void setUp() throws Exception {
        super.setUp();
        app = getInstrumentation().getTargetContext();
        environment = OrtEnvironment.getEnvironment();
    }

    public void testLabelOrder() throws Exception {
        String[] expected = {"bottles", "cups", "pens", "scissors"};
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(app.getAssets().open("labels.txt")))) {
            for (String label : expected) assertEquals(label, reader.readLine().trim().toLowerCase(Locale.US));
            assertNull("Expected exactly four labels", reader.readLine());
        }
    }

    public void testLetterboxPreservesAspectAndGrayPadding() {
        Bitmap original = Bitmap.createBitmap(320, 160, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(original); canvas.drawColor(Color.RED);
        Bitmap prepared = MainActivity.makeModelImage(original);
        assertEquals(224, prepared.getWidth()); assertEquals(224, prepared.getHeight());
        int pad = prepared.getPixel(112, 4);
        assertEquals(127, Color.red(pad)); assertEquals(127, Color.green(pad)); assertEquals(127, Color.blue(pad));
        int center = prepared.getPixel(112, 112);
        assertTrue(Color.red(center) > 245); assertTrue(Color.green(center) < 10); assertTrue(Color.blue(center) < 10);
        float[] chw = MainActivity.preprocess(prepared);
        assertEquals(3 * 224 * 224, chw.length);
        assertEquals(127f / 255f, chw[0], .01f);
        int centerIndex = 112 * 224 + 112;
        assertEquals(1f, chw[centerIndex], .01f);
        assertEquals(0f, chw[224 * 224 + centerIndex], .01f);
    }

    public void testBothModelsReturnNormalizedFourClassProbabilitiesAndHeatmaps() throws Exception {
        Bitmap sample = findSampleOrFallback();
        Bitmap prepared = MainActivity.makeModelImage(sample);
        float[] pixels = MainActivity.preprocess(prepared);
        float[] batchPixels = new float[pixels.length * 2];
        System.arraycopy(pixels, 0, batchPixels, 0, pixels.length);
        System.arraycopy(pixels, 0, batchPixels, pixels.length, pixels.length);
        for (String model : MODELS) {
            try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                options.setIntraOpNumThreads(2);
                try (OrtSession session = environment.createSession(copyAssetToCache(model), options)) {
                    assertTrue(session.getInputNames().contains("images"));
                    assertTrue(session.getOutputNames().contains("probabilities"));
                    assertTrue(session.getOutputNames().contains("heatmaps"));
                    try (OnnxTensor tensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(batchPixels), new long[]{2,3,224,224});
                         OrtSession.Result result = session.run(Collections.singletonMap("images", tensor))) {
                        float[][] probabilities = readProbabilityBatch(result.get("probabilities").get().getValue());
                        assertEquals("dynamic batch dimension", 2, probabilities.length);
                        for (float[] probability : probabilities) {
                            assertEquals(4, probability.length); float sum=0;
                            for (float value : probability) { assertFalse(Float.isNaN(value)); assertFalse(Float.isInfinite(value)); assertTrue(value >= 0 && value <= 1); sum += value; }
                            assertEquals("probabilities must sum to one", 1f, sum, .02f);
                        }
                        Object value = result.get("heatmaps").get().getValue();
                        assertTrue("Expected float32 heatmaps", value instanceof float[][][][]);
                        float[][][][] maps = (float[][][][]) value;
                        assertEquals("dynamic batch dimension", 2, maps.length); assertEquals(4, maps[0].length); assertEquals(4, maps[1].length);
                        assertTrue(maps[0][0].length > 0); assertTrue(maps[0][0][0].length > 0);
                        for (float[][][] batch : maps) for (float[][] byClass : batch) for (float[] line : byClass) for (float h : line) {
                            assertFalse(Float.isNaN(h)); assertFalse(Float.isInfinite(h)); assertTrue(h >= 0 && h <= 1);
                        }
                    }
                }
            }
        }
    }
    public void testHeatmapToggleAndModelSwitchClearResult() throws Exception {
        Instrumentation instrumentation = getInstrumentation();
        Intent launch = new Intent(app, MainActivity.class); launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        final MainActivity activity = (MainActivity) instrumentation.startActivitySync(launch);
        instrumentation.waitForIdleSync();
        try {
            final Button heat = (Button) field(activity, "heatmapButton").get(activity);
            final TextView prediction = (TextView) field(activity, "prediction").get(activity);
            final boolean[] state = new boolean[2];
            Bitmap testImage = MainActivity.makeModelImage(Bitmap.createBitmap(224,224,Bitmap.Config.ARGB_8888));
            field(activity,"modelImage").set(activity,testImage);
            float[][][] maps = new float[4][2][2]; maps[0][0][0]=1f;
            instrumentation.runOnMainSync(() -> {
                activity.render(new float[]{.7f,.1f,.1f,.1f}, maps, 1, 0);
                state[0] = heat.isEnabled() && "Bottles".contentEquals(prediction.getText());
                heat.performClick();
                try { state[1] = (boolean) field(activity,"showHeatmap").get(activity); } catch(Exception e){ throw new RuntimeException(e); }
                activity.selectModel(1);
            });
            assertTrue("Heatmap becomes available after inference", state[0]);
            assertTrue("Toggle displays the heatmap", state[1]);
            assertFalse("Switching model clears prior heatmap", heat.isEnabled());
            assertEquals("", prediction.getText().toString());
        } finally { instrumentation.runOnMainSync(activity::finish); instrumentation.waitForIdleSync(); }
    }



    public void testArtAugGalleryCyclesFourClassesAndWraps() throws Exception {
        Instrumentation instrumentation=getInstrumentation();
        Intent launch=new Intent(app,MainActivity.class);launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        final MainActivity activity=(MainActivity)instrumentation.startActivitySync(launch);
        instrumentation.waitForIdleSync();
        try {
            ArtAugView panel=(ArtAugView)field(activity,"artAugView").get(activity);
            final Button generate=(Button)field(panel,"generateButton").get(panel);
            final TextView classTitle=(TextView)field(panel,"classTitle").get(panel);
            final ImageView original=(ImageView)field(panel,"originalImage").get(panel);
            final ImageView enhanced=(ImageView)field(panel,"enhancedImage").get(panel);
            final String[] seen=new String[5];
            String[] expected={"Bottles","Cups","Pens","Scissors","Bottles"};
            assertTrue("Four reviewed pairs should enable the gallery",generate.isEnabled());
            instrumentation.runOnMainSync(()->{
                activity.selectTab(2);
            });
            for(int i=0;i<5;i++){
                waitForEnabled(instrumentation,generate);
                instrumentation.runOnMainSync(()->generate.performClick());
                instrumentation.waitForIdleSync();
                final int slot=i;
                instrumentation.runOnMainSync(()->seen[slot]=classTitle.getText().toString()+"|"+panel.getSelectedIndex());
                assertEquals(expected[i]+" · "+((i%4)+1)+"/4|"+(i%4),seen[i]);
            }
            waitForEnabled(instrumentation,generate);
            assertEquals("Show the next saved offline ArtAug photo",generate.getContentDescription().toString());
            assertEquals("Generate",generate.getText().toString());
            assertEquals("Original photo",original.getContentDescription().toString());
            assertEquals("ArtAug photo enhanced by AI",enhanced.getContentDescription().toString());
            assertEquals(1,panel.getClassCursor());
            assertTrue("Original image must appear above its enhanced pair",panel.indexOfChild(original)<panel.indexOfChild(enhanced));
            TextView originalCaption=(TextView)field(panel,"originalCaption").get(panel);
            TextView enhancedCaption=(TextView)field(panel,"enhancedCaption").get(panel);
            assertEquals("ORIGINAL",originalCaption.getText().toString());
            assertEquals("ENHANCED BY AI",enhancedCaption.getText().toString());
            assertEquals(View.VISIBLE,original.getVisibility());
            assertEquals(View.VISIBLE,originalCaption.getVisibility());
            assertEquals(View.VISIBLE,enhanced.getVisibility());
            assertEquals(View.VISIBLE,enhancedCaption.getVisibility());
            assertTrue("Original caption should follow source image",panel.indexOfChild(original)<panel.indexOfChild(originalCaption));
            assertTrue("Enhanced image should follow original caption",panel.indexOfChild(originalCaption)<panel.indexOfChild(enhanced));
            assertTrue("Enhanced caption should follow enhanced image",panel.indexOfChild(enhanced)<panel.indexOfChild(enhancedCaption));
        } finally {instrumentation.runOnMainSync(activity::finish);instrumentation.waitForIdleSync();}
    }

    public void testGeneratedPhotosTabKeepsBothClassifierTabsAvailable() throws Exception {
        Instrumentation instrumentation=getInstrumentation();
        Intent launch=new Intent(app,MainActivity.class);launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        final MainActivity activity=(MainActivity)instrumentation.startActivitySync(launch);
        instrumentation.waitForIdleSync();
        try {
            final boolean[] shown=new boolean[1];
            instrumentation.runOnMainSync(()->{
                try {
                    android.widget.LinearLayout panel=(android.widget.LinearLayout)field(activity,"artAugView").get(activity);
                    TextView eyebrow=(TextView)field(activity,"appEyebrow").get(activity);
                    android.widget.LinearLayout classifier=(android.widget.LinearLayout)field(activity,"classifierPanel").get(activity);
                    TextView[] tabs=(TextView[])field(activity,"tabViews").get(activity);
                    boolean tabNames="Baseline".contentEquals(tabs[0].getText()) && "Enchanced Model".contentEquals(tabs[1].getText()) && "Generated Photos".contentEquals(tabs[2].getText());
                    activity.selectTab(0);
                    boolean baseline=tabNames && classifier.getVisibility()==View.VISIBLE && "Baseline classifier".contentEquals(tabs[0].getContentDescription());
                    boolean baselineSelected=(int)field(activity,"selectedModel").get(activity)==0;
                    activity.selectTab(1);
                    boolean second=classifier.getVisibility()==View.VISIBLE && "Enchanced Model classifier".contentEquals(tabs[1].getContentDescription());
                    boolean secondSelected=(int)field(activity,"selectedModel").get(activity)==1;
                    activity.selectTab(2);
                    shown[0]=baseline && second && baselineSelected && secondSelected && panel.getVisibility()==View.VISIBLE && eyebrow.getVisibility()==View.GONE;
                } catch(Exception e){throw new RuntimeException(e);}
            });
            assertTrue("Third tab opens the ArtAug offline review panel",shown[0]);
            Button generate=(Button)field(field(activity,"artAugView").get(activity),"generateButton").get(field(activity,"artAugView").get(activity));
            assertTrue("Packaged approved pairs should enable Generate",generate.isEnabled());
        } finally {instrumentation.runOnMainSync(activity::finish);instrumentation.waitForIdleSync();}
    }

    public void testArtAugPackagedPairsHaveAuthenticJpegsAndHashes() throws Exception {
        String[] expected={"bottles","cups","pens","scissors"};
        JSONObject root;
        try(InputStream in=app.getAssets().open("artaug/manifest.json")) {
            java.io.ByteArrayOutputStream out=new java.io.ByteArrayOutputStream();byte[] b=new byte[8192];int n;
            while((n=in.read(b))>0)out.write(b,0,n);
            root=new JSONObject(new String(out.toByteArray(),java.nio.charset.StandardCharsets.UTF_8));
        }
        JSONArray entries=root.getJSONArray("examples");
        assertEquals("Exactly four reviewed examples are packaged",4,entries.length());
        assertTrue("Manifest identifies these as synthetic released-adapter images",root.getJSONObject("generator").getBoolean("synthetic_images"));
        for(int i=0;i<entries.length();i++){
            JSONObject entry=entries.getJSONObject(i);
            assertTrue(entry.getBoolean("approved"));
            assertEquals(expected[i],entry.getString("class_name"));
            for(String kind:new String[]{"original","enhanced"}){
                byte[] bytes;
                try(InputStream image=app.getAssets().open("artaug/"+entry.getString(kind))){
                    java.io.ByteArrayOutputStream out=new java.io.ByteArrayOutputStream();byte[] b=new byte[8192];int n;
                    while((n=image.read(b))>0)out.write(b,0,n);bytes=out.toByteArray();
                }
                assertTrue(kind+" must be JPEG",bytes.length>3&&bytes[0]==(byte)0xff&&bytes[1]==(byte)0xd8&&bytes[2]==(byte)0xff);
                java.security.MessageDigest digest=java.security.MessageDigest.getInstance("SHA-256");
                StringBuilder hex=new StringBuilder();for(byte value:digest.digest(bytes))hex.append(String.format(java.util.Locale.ROOT,"%02x",value&0xff));
                assertEquals(kind+" checksum",entry.getString(kind+"_sha256"),hex.toString());
            }
        }
    }

    private Bitmap findSampleOrFallback() throws Exception {
        String[] samples = app.getAssets().list("samples");
        if (samples != null && samples.length > 0) {
            try (InputStream stream = app.getAssets().open("samples/" + samples[0])) {
                Bitmap bitmap = android.graphics.BitmapFactory.decodeStream(stream);
                if (bitmap != null) return bitmap;
            }
        }
        Bitmap fallback=Bitmap.createBitmap(320,160,Bitmap.Config.ARGB_8888); Canvas canvas=new Canvas(fallback); Paint paint=new Paint();
        canvas.drawColor(Color.rgb(127,127,127)); paint.setColor(Color.WHITE); canvas.drawRect(70,25,150,140,paint); return fallback;
    }

    private String copyAssetToCache(String file) throws Exception {
        java.io.File target = new java.io.File(app.getCacheDir(), file);
        try (InputStream in = app.getAssets().open(file); java.io.FileOutputStream out = new java.io.FileOutputStream(target)) {
            byte[] buffer = new byte[16384]; int n; while ((n=in.read(buffer))>0) out.write(buffer,0,n);
        }
        return target.getAbsolutePath();
    }
    private float[][] readProbabilityBatch(Object value) {
        if(value instanceof float[][]) return (float[][])value;
        if(value instanceof float[]) return new float[][]{(float[])value};
        fail("Unexpected probabilities shape: "+value.getClass()); return null;
    }
    private void waitForEnabled(Instrumentation instrumentation,Button button) {
        long deadline=android.os.SystemClock.elapsedRealtime()+8000;
        boolean[] enabled=new boolean[1];
        while(android.os.SystemClock.elapsedRealtime()<deadline){
            instrumentation.waitForIdleSync();
            instrumentation.runOnMainSync(()->enabled[0]=button.isEnabled());
            if(enabled[0])return;
            android.os.SystemClock.sleep(25);
        }
        fail("Gallery Generate button did not become enabled after loading the saved photo");
    }
    private Field field(Object target,String name) throws Exception { Field f=target.getClass().getDeclaredField(name); f.setAccessible(true); return f; }
}







