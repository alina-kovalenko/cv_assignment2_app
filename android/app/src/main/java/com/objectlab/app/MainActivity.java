package com.objectlab.app;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.ClipData;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Bundle;
import android.os.Build;
import android.provider.MediaStore;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.Space;
import android.widget.TextView;
import android.widget.Toast;

import androidx.core.content.FileProvider;
import androidx.core.view.WindowCompat;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.exifinterface.media.ExifInterface;

import java.io.File;
import java.io.FileOutputStream;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

/** Offline four-class photo classifier. Model work is serialized on one worker. */
public final class MainActivity extends Activity {
    private static final int REQUEST_GALLERY = 1;
    private static final int REQUEST_CAMERA = 2;
    private static final String STATE_IMAGE = "selected_image";
    private static final String STATE_TAB = "selected_tab";
    private int selectedTab = 0;
    private LinearLayout classifierPanel;
    private ArtAugView artAugView;
    private TextView appEyebrow, appHeadline;
    private static final String[] CLASSES = {"Bottles", "Cups", "Pens", "Scissors"};
    private static final String[] MODEL_FILES = {"baseline.onnx", "sota.onnx"};
    private static final int INPUT_SIZE = 224;
    private static final int BG = Color.rgb(8, 8, 8), PANEL = Color.rgb(20, 20, 20);
    private static final int WHITE = Color.WHITE, MUTED = Color.rgb(160, 160, 160);

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final OrtSession[] sessions = new OrtSession[2];
    private final double[] averageMs = new double[2];
    private final int[] runCount = new int[2];
    private OrtEnvironment environment;
    private OrtSession.SessionOptions sessionOptions;

    private int selectedModel = 0;
    private long generation = 0;
    private boolean busy = false, closing = false, showHeatmap = false;
    private Uri selectedUri, cameraUri;
    private String selectedSample;
    private Bitmap modelImage, selectedHeatmap;
    private float[] selectedProbabilities;

    private ImageView preview;
    private TextView status, prediction, confidence, latency, emptyState, modelDescription;
    private LinearLayout heatmapLegend;
    private LinearLayout resultRows;
    private Button runButton, heatmapButton, sampleButton;
    private TextView[] tabViews;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().setStatusBarColor(BG);
        getWindow().setNavigationBarColor(BG);
        WindowCompat.setDecorFitsSystemWindows(getWindow(), false);
        if (state != null) {
            selectedModel = state.getInt("selected_model", 0);
            selectedTab = state.getInt(STATE_TAB, 0);
            String camera = state.getString("camera_uri"); if(camera!=null) cameraUri=Uri.parse(camera);
            String uri = state.getString(STATE_IMAGE);
            if (uri != null) selectedUri = Uri.parse(uri);
            selectedSample = state.getString("selected_sample");
        }
        buildUi();
        if (state != null) artAugView.setSelectionState(state.getInt("artaug_index", -1), state.getInt("artaug_class_cursor", 0));
        updateModelUi();
        selectTab(selectedTab);
        if (selectedSample != null) loadSample(selectedSample); else if (selectedUri != null) loadUri(selectedUri, false);
        loadModels();
    }

    @Override protected void onSaveInstanceState(Bundle out) {
        out.putInt("selected_model", selectedModel);
        out.putInt(STATE_TAB, selectedTab);
        if (artAugView != null) {
            out.putInt("artaug_index", artAugView.getSelectedIndex());
            out.putInt("artaug_class_cursor", artAugView.getClassCursor());
        }
        if (selectedUri != null) out.putString(STATE_IMAGE, selectedUri.toString());
        if (cameraUri != null) out.putString("camera_uri", cameraUri.toString());
        if (selectedSample != null) out.putString("selected_sample", selectedSample);
        super.onSaveInstanceState(out);
    }

    private int dp(float value) { return (int)(value * getResources().getDisplayMetrics().density + .5f); }
    private LinearLayout column() { LinearLayout v = new LinearLayout(this); v.setOrientation(LinearLayout.VERTICAL); return v; }
    private LinearLayout row() { LinearLayout v = new LinearLayout(this); v.setOrientation(LinearLayout.HORIZONTAL); v.setGravity(Gravity.CENTER_VERTICAL); return v; }
    private TextView label(String value, int size, int color, boolean bold) {
        TextView t = new TextView(this); t.setText(value); t.setTextSize(size); t.setTextColor(color);
        if (bold) t.setTypeface(null, android.graphics.Typeface.BOLD);
        return t;
    }
    private android.graphics.drawable.GradientDrawable shape(int color, int radius, int stroke) {
        android.graphics.drawable.GradientDrawable d = new android.graphics.drawable.GradientDrawable();
        d.setColor(color); d.setCornerRadius(dp(radius)); if (stroke != 0) d.setStroke(dp(1), stroke); return d;
    }

    private void buildUi() {
        ScrollView scroll = new ScrollView(this); scroll.setFillViewport(true);
        ViewCompat.setOnApplyWindowInsetsListener(scroll, (view, insets) -> { androidx.core.graphics.Insets bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() | WindowInsetsCompat.Type.displayCutout()); view.setPadding(bars.left, bars.top, bars.right, bars.bottom); return insets; });
        LinearLayout root = column(); root.setPadding(dp(20), dp(14), dp(20), dp(28)); root.setBackgroundColor(BG); scroll.addView(root); setContentView(scroll);
        LinearLayout top = row(); TextView brand = label("OBJECTLAB", 12, WHITE, true); brand.setLetterSpacing(.16f); top.addView(brand);
        Space flex = new Space(this); top.addView(flex, new LinearLayout.LayoutParams(0, 1, 1));
        TextView about = label("ⓘ", 22, MUTED, false); about.setContentDescription("About this app"); about.setMinWidth(dp(48)); about.setMinHeight(dp(48)); about.setGravity(Gravity.CENTER); about.setOnClickListener(v -> showAbout()); top.addView(about); root.addView(top);
        appEyebrow = label("HOUSEHOLD OBJECT CLASSIFIER", 10, MUTED, true); appEyebrow.setLetterSpacing(.12f);
        LinearLayout.LayoutParams ep = new LinearLayout.LayoutParams(-2, -2); ep.topMargin = dp(10); root.addView(appEyebrow, ep);
        appHeadline = label("See what\nyou captured.", 34, WHITE, true); LinearLayout.LayoutParams ht = new LinearLayout.LayoutParams(-1, -2); ht.topMargin = dp(16); root.addView(appHeadline, ht);

        LinearLayout tabs = row(); tabs.setPadding(dp(4), dp(4), dp(4), dp(4)); tabs.setBackground(shape(PANEL, 14, 0)); tabViews = new TextView[3];
        String[] tabNames = {"Baseline", "Enchanced Model", "Generated Photos"};
        for (int i = 0; i < 3; i++) { final int index = i; TextView t = label(tabNames[i], 11, WHITE, true); t.setGravity(Gravity.CENTER); t.setPadding(dp(2), dp(10), dp(2), dp(10)); t.setMinHeight(dp(48)); t.setContentDescription(i==2?"Generated Photos, offline ArtAug gallery":(i==0?"Baseline classifier":"Enchanced Model classifier")); t.setOnClickListener(v -> { if(index<2)selectModel(index); selectTab(index); }); tabViews[i] = t; tabs.addView(t, new LinearLayout.LayoutParams(0, -2, 1)); }
        LinearLayout.LayoutParams tp = new LinearLayout.LayoutParams(-1, -2); tp.topMargin = dp(22); root.addView(tabs, tp);
        classifierPanel = column(); root.addView(classifierPanel);
        modelDescription = label("", 10, MUTED, true); modelDescription.setLetterSpacing(.07f); LinearLayout.LayoutParams md = new LinearLayout.LayoutParams(-2, -2); md.topMargin = dp(13); classifierPanel.addView(modelDescription, md);
        preview = new ImageView(this); preview.setScaleType(ImageView.ScaleType.FIT_CENTER); preview.setBackground(shape(PANEL, 18, Color.rgb(42, 42, 42))); preview.setClipToOutline(true); preview.setContentDescription("Selected photo preview");
        LinearLayout.LayoutParams pp = new LinearLayout.LayoutParams(-1, dp(285)); pp.topMargin = dp(16); classifierPanel.addView(preview, pp);
        emptyState = label("Choose a photo to begin", 14, MUTED, true); emptyState.setGravity(Gravity.CENTER); emptyState.setMinHeight(dp(48)); classifierPanel.addView(emptyState); heatmapLegend = column(); TextView legendTitle = label("MODEL ATTENTION", 9, MUTED, true); legendTitle.setLetterSpacing(.1f); heatmapLegend.addView(legendTitle); LinearLayout legendRow = row(); legendRow.addView(label("LOW", 9, MUTED, true)); View gradient = new View(this); android.graphics.drawable.GradientDrawable gradientDrawable = new android.graphics.drawable.GradientDrawable(android.graphics.drawable.GradientDrawable.Orientation.LEFT_RIGHT, new int[]{Color.rgb(45,105,220), Color.rgb(35,190,200), Color.rgb(248,210,65), Color.rgb(235,75,55)}); gradientDrawable.setCornerRadius(dp(4)); gradient.setBackground(gradientDrawable); LinearLayout.LayoutParams gp = new LinearLayout.LayoutParams(0, dp(7), 1); gp.leftMargin = dp(8); gp.rightMargin = dp(8); legendRow.addView(gradient, gp); legendRow.addView(label("HIGH", 9, MUTED, true)); heatmapLegend.addView(legendRow); heatmapLegend.setVisibility(View.GONE); classifierPanel.addView(heatmapLegend); preview.setOnClickListener(v -> chooseGallery());
        LinearLayout actions = row(); LinearLayout.LayoutParams ap = new LinearLayout.LayoutParams(-1, -2); ap.topMargin = dp(12); classifierPanel.addView(actions, ap);
        Button gallery = makeButton("Choose photo", false); gallery.setContentDescription("Choose a photo from your gallery"); gallery.setOnClickListener(v -> chooseGallery());
        Button camera = makeButton("Camera", false); camera.setContentDescription("Take a full resolution photo with the camera"); camera.setOnClickListener(v -> takePhoto());
        actions.addView(gallery, new LinearLayout.LayoutParams(0, dp(52), 1)); LinearLayout.LayoutParams cp = new LinearLayout.LayoutParams(0, dp(52), 1); cp.leftMargin = dp(10); actions.addView(camera, cp);
        sampleButton = makeButton("Try a sample photo", false); sampleButton.setContentDescription("Choose an included sample photo"); sampleButton.setVisibility(View.GONE); sampleButton.setOnClickListener(v -> showSamples()); LinearLayout.LayoutParams sbp = new LinearLayout.LayoutParams(-1, dp(48)); sbp.topMargin = dp(8); classifierPanel.addView(sampleButton, sbp); updateSampleButton();
        runButton = makeButton("Run inference  →", true); runButton.setEnabled(false); runButton.setOnClickListener(v -> runInference()); LinearLayout.LayoutParams rp = new LinearLayout.LayoutParams(-1, dp(52)); rp.topMargin = dp(12); classifierPanel.addView(runButton, rp);
        status = label("LOADING MODELS · OFFLINE", 10, MUTED, true); LinearLayout.LayoutParams sp = new LinearLayout.LayoutParams(-2, -2); sp.topMargin = dp(16); classifierPanel.addView(status, sp);
        LinearLayout resultHead = row(); TextView rh = label("PREDICTION", 11, MUTED, true); rh.setLetterSpacing(.12f); resultHead.addView(rh); Space spacer = new Space(this); resultHead.addView(spacer, new LinearLayout.LayoutParams(0, 1, 1));
        heatmapButton = makeButton("Show heatmap", false); heatmapButton.setTextSize(11); heatmapButton.setPadding(dp(10), dp(6), dp(10), dp(6)); heatmapButton.setEnabled(false); heatmapButton.setOnClickListener(v -> toggleHeatmap()); resultHead.addView(heatmapButton);
        LinearLayout.LayoutParams rhl = new LinearLayout.LayoutParams(-1, -2); rhl.topMargin = dp(24); classifierPanel.addView(resultHead, rhl);
        prediction = label("", 26, WHITE, true); classifierPanel.addView(prediction); confidence = label("", 12, MUTED, false); classifierPanel.addView(confidence);
        resultRows = column(); classifierPanel.addView(resultRows);
        latency = label("Latency · —", 11, MUTED, false); LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-1, -2); lp.topMargin = dp(16); classifierPanel.addView(latency, lp);
        TextView note = label("Only four known classes are supported. Confidence is not certainty; unfamiliar objects can be misclassified.", 12, MUTED, false); note.setLineSpacing(dp(3), 1); LinearLayout.LayoutParams np = new LinearLayout.LayoutParams(-1, -2); np.topMargin = dp(22); classifierPanel.addView(note, np);
        artAugView = new ArtAugView(this, worker);
        root.addView(artAugView);
    }

    private Button makeButton(String text, boolean primary) {
        Button b = new Button(this); b.setText(text); b.setAllCaps(false); b.setTextSize(14); b.setTypeface(null, android.graphics.Typeface.BOLD); b.setMinHeight(dp(48));
        b.setTextColor(WHITE); b.setBackground(shape(primary ? Color.rgb(245,245,245) : PANEL, 13, primary ? 0 : Color.rgb(55,55,55))); if (primary) b.setTextColor(Color.BLACK); return b;
    }
    private void updateSampleButton() { try { String[] files=getAssets().list("samples"); sampleButton.setVisibility(files!=null && files.length>0 ? View.VISIBLE : View.GONE); } catch(Exception ignored) { sampleButton.setVisibility(View.GONE); } }
    private void showSamples() { try { String[] files=getAssets().list("samples"); if(files==null||files.length==0){toast("No sample photos included");return;} new AlertDialog.Builder(this).setTitle("Sample photos").setItems(files,(dialog,index)->loadSample(files[index])).show(); } catch(Exception e){toast("Samples are unavailable");} }
    private void loadSample(String name) {
        releaseSelectedGalleryPermission(); if(isCameraUri(selectedUri))deleteCameraUri(selectedUri); cameraUri=null; generation++; selectedSample=name; selectedUri=null; modelImage=null; clearPrediction(); status.setText("LOADING SAMPLE…"); emptyState.setVisibility(View.VISIBLE); updateRunAvailability(); final long token=generation;
        worker.execute(()->{try(InputStream in=getAssets().open("samples/"+name); java.io.ByteArrayOutputStream bytes=new java.io.ByteArrayOutputStream()){
            byte[] buffer=new byte[8192]; int count; while((count=in.read(buffer))>0)bytes.write(buffer,0,count); byte[] imageData=bytes.toByteArray();
            BitmapFactory.Options bounds=new BitmapFactory.Options(); bounds.inJustDecodeBounds=true; BitmapFactory.decodeByteArray(imageData,0,imageData.length,bounds);
            int max=Math.max(bounds.outWidth,bounds.outHeight),sample=1; while(max/sample>1600)sample*=2; BitmapFactory.Options opts=new BitmapFactory.Options(); opts.inSampleSize=sample;
            Bitmap decodedImage=BitmapFactory.decodeByteArray(imageData,0,imageData.length,opts); if(decodedImage==null)throw new IllegalArgumentException("Bad sample image"); int orientation; try(InputStream exif=new ByteArrayInputStream(imageData)){orientation=new ExifInterface(exif).getAttributeInt(ExifInterface.TAG_ORIENTATION,ExifInterface.ORIENTATION_NORMAL);} Bitmap orientedImage=applyExifOrientation(decodedImage,orientation); final Bitmap padded=makeModelImage(orientedImage);
            runOnUiThread(()->{if(closing||token!=generation)return;selectedUri=null;modelImage=padded;preview.setImageBitmap(padded);emptyState.setVisibility(View.GONE);status.setText("SAMPLE PHOTO · IMAGE STAYS ON DEVICE");updateRunAvailability();});
        }catch(Exception e){runOnUiThread(()->{if(token==generation&&!closing){status.setText("COULD NOT OPEN SAMPLE");toast("Sample photo could not be opened");}});}});
    }
    private void showAbout() {
        new AlertDialog.Builder(this).setTitle("About ObjectLab")
                .setMessage("Baseline is the retrained A1 CNN. Enchanced Model uses MobileNetV4 transfer learning. Both classifiers run on this device without uploading images. They recognize bottles, cups, pens, and scissors only; confidence is not certainty. Generated Photos shows saved ArtAug pairs created offline.")
                .setPositiveButton("Done", null).show();
    }
    void selectModel(int index) {
        if (index == selectedModel) return;
        selectedModel = index; generation++; clearPrediction(); updateModelUi(); updateRunAvailability();
    }
    void selectTab(int index) {
        int previousTab=selectedTab;
        selectedTab=Math.max(0,Math.min(2,index));
        if(previousTab==2&&selectedTab!=2)artAugView.onTabHidden();
        if(previousTab!=2&&selectedTab==2)artAugView.onTabShown();
        appEyebrow.setVisibility(selectedTab==2?View.GONE:View.VISIBLE);
        appHeadline.setVisibility(selectedTab==2?View.GONE:View.VISIBLE);
        appEyebrow.setText("HOUSEHOLD OBJECT CLASSIFIER");
        appHeadline.setText("See what\nyou captured.");
        classifierPanel.setVisibility(selectedTab==2?View.GONE:View.VISIBLE);
        artAugView.setVisibility(selectedTab==2?View.VISIBLE:View.GONE);
        if(selectedTab<2 && selectedTab!=selectedModel){selectedModel=selectedTab;generation++;clearPrediction();updateRunAvailability();}
        for(int i=0;i<3;i++) tabViews[i].setBackground(shape(i==selectedTab?Color.rgb(43,43,43):Color.TRANSPARENT,11,0));
    }
    private void updateModelUi() {
        for (int i=0;i<3;i++) tabViews[i].setBackground(shape(i==selectedTab ? Color.rgb(43,43,43) : Color.TRANSPARENT, 11, 0));
        modelDescription.setText(selectedModel == 0 ? "RETRAINED A1 CNN · 4 CLASSES" : "MOBILENETV4 · TRANSFER LEARNING");
    }
    private void chooseGallery() { Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT); i.addCategory(Intent.CATEGORY_OPENABLE); i.setType("image/*"); startActivityForResult(i, REQUEST_GALLERY); }
    private void takePhoto() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (intent.resolveActivity(getPackageManager()) == null) { toast("No camera app is available"); return; }
        try {
            File dir = new File(getCacheDir(), "camera"); if (!dir.exists() && !dir.mkdirs()) throw new IllegalStateException("Cannot create camera cache");
            File file = File.createTempFile("objectlab_", ".jpg", dir);
            cameraUri = FileProvider.getUriForFile(this, getPackageName()+".fileprovider", file);
            intent.putExtra(MediaStore.EXTRA_OUTPUT, cameraUri);
            intent.setClipData(ClipData.newUri(getContentResolver(), "ObjectLab camera output", cameraUri));
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
            startActivityForResult(intent, REQUEST_CAMERA);
        } catch (Exception e) {
            if (cameraUri != null && !cameraUri.equals(selectedUri)) deleteCameraUri(cameraUri);
            cameraUri = null;
            toast("Could not open camera");
        }
    }
    @Override protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (request == REQUEST_CAMERA) {
            if (result != RESULT_OK || cameraUri == null) {
                if (cameraUri != null && !cameraUri.equals(selectedUri)) deleteCameraUri(cameraUri);
                cameraUri = null;
                return;
            }
            Uri capturedUri = cameraUri;
            cameraUri = null;
            loadUri(capturedUri, true);
            return;
        }
        if (request == REQUEST_GALLERY && result == RESULT_OK && data != null && data.getData() != null) {
            try { getContentResolver().takePersistableUriPermission(data.getData(), Intent.FLAG_GRANT_READ_URI_PERMISSION); } catch (SecurityException ignored) {}
            loadUri(data.getData(), true);
        }
    }
    private void loadUri(Uri uri, boolean announce) {
        if (selectedUri != null && !selectedUri.equals(uri)) { releaseGalleryPermission(selectedUri); if(isCameraUri(selectedUri))deleteCameraUri(selectedUri); } generation++; selectedSample = null; selectedUri = uri; modelImage = null; preview.setImageDrawable(null); emptyState.setVisibility(View.VISIBLE); clearPrediction(); showHeatmap = false; updateHeatmapButton(); status.setText("LOADING PHOTO…"); updateRunAvailability();
        final long token = generation;
        worker.execute(() -> {
            try {
                Bitmap decoded = decode(uri); Bitmap padded = makeModelImage(decoded);
                runOnUiThread(() -> { if (closing || token != generation) return; modelImage = padded; preview.setImageBitmap(padded); preview.setContentDescription("Selected photo, shown with gray padding matching the model input"); emptyState.setVisibility(View.GONE); status.setText("PHOTO READY · IMAGE STAYS ON DEVICE"); updateRunAvailability(); });
            } catch (Exception e) { runOnUiThread(() -> { if (token != generation || closing) return; releaseGalleryPermission(selectedUri); if(isCameraUri(selectedUri))deleteCameraUri(selectedUri); selectedUri = null; modelImage = null; preview.setImageDrawable(null); emptyState.setVisibility(View.VISIBLE); status.setText("COULD NOT OPEN PHOTO"); toast("Could not decode this image"); updateRunAvailability(); }); }
        });
    }
    private boolean isCameraUri(Uri uri) { return uri != null && (getPackageName()+".fileprovider").equals(uri.getAuthority()); }
    private void deleteCameraUri(Uri uri) {
        if (!isCameraUri(uri)) return;
        java.util.List<String> segments=uri.getPathSegments();
        if (segments.size()!=2 || !"camera".equals(segments.get(0))) return;
        try {
            File directory=new File(getCacheDir(),"camera").getCanonicalFile();
            File target=new File(directory,segments.get(1)).getCanonicalFile();
            if (target.getParentFile()!=null && target.getParentFile().equals(directory)) target.delete();
        } catch (Exception ignored) {}
    }
    private void releaseSelectedGalleryPermission() { releaseGalleryPermission(selectedUri); }
    private void releaseGalleryPermission(Uri uri) { if (uri == null || (getPackageName()+".fileprovider").equals(uri.getAuthority())) return; try { getContentResolver().releasePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION); } catch (SecurityException ignored) {} }
    private Bitmap decode(Uri uri) throws Exception {
        BitmapFactory.Options bounds = new BitmapFactory.Options(); bounds.inJustDecodeBounds = true;
        try (InputStream in = getContentResolver().openInputStream(uri)) { if (in == null) throw new IllegalArgumentException("Photo unavailable"); BitmapFactory.decodeStream(in, null, bounds); }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) throw new IllegalArgumentException("Invalid image dimensions");
        int max = Math.max(bounds.outWidth, bounds.outHeight); int sample = 1; while (max / sample > 1600) sample *= 2;
        BitmapFactory.Options opts = new BitmapFactory.Options(); opts.inSampleSize = sample;
        Bitmap bitmap; try (InputStream in = getContentResolver().openInputStream(uri)) { if (in == null) throw new IllegalArgumentException("Photo unavailable"); bitmap = BitmapFactory.decodeStream(in, null, opts); }
        if (bitmap == null) throw new IllegalArgumentException("Unsupported image format");
        int orientation = ExifInterface.ORIENTATION_NORMAL;
        try (InputStream in = getContentResolver().openInputStream(uri)) { if (in != null) orientation = new ExifInterface(in).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL); }
        return applyExifOrientation(bitmap, orientation);
    }
    static Bitmap applyExifOrientation(Bitmap bitmap, int orientation) {
        if (orientation == ExifInterface.ORIENTATION_NORMAL || orientation == ExifInterface.ORIENTATION_UNDEFINED) return bitmap;
        Matrix matrix = new Matrix();
        switch (orientation) {
            case ExifInterface.ORIENTATION_FLIP_HORIZONTAL: matrix.setScale(-1, 1); break;
            case ExifInterface.ORIENTATION_ROTATE_180: matrix.setRotate(180); break;
            case ExifInterface.ORIENTATION_FLIP_VERTICAL: matrix.setScale(1, -1); break;
            case ExifInterface.ORIENTATION_TRANSPOSE: matrix.setRotate(90); matrix.postScale(-1, 1); break;
            case ExifInterface.ORIENTATION_ROTATE_90: matrix.setRotate(90); break;
            case ExifInterface.ORIENTATION_TRANSVERSE: matrix.setRotate(-90); matrix.postScale(-1, 1); break;
            case ExifInterface.ORIENTATION_ROTATE_270: matrix.setRotate(-90); break;
            default: return bitmap;
        }
        return Bitmap.createBitmap(bitmap,0,0,bitmap.getWidth(),bitmap.getHeight(),matrix,true);
    }
    /**
     * Model and preview share the same square canvas and integer-centered padding as Pillow
     * ImageOps.pad. Canvas filtering is bilinear; Pillow training used Lanczos.
     */
    static Bitmap makeModelImage(Bitmap source) {
        Bitmap output = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(output);
        canvas.drawColor(Color.rgb(127,127,127));
        double scale = Math.min((double)INPUT_SIZE/source.getWidth(), (double)INPUT_SIZE/source.getHeight());
        int scaledWidth = Math.max(1, (int)Math.rint(source.getWidth()*scale));
        int scaledHeight = Math.max(1, (int)Math.rint(source.getHeight()*scale));
        int left = (INPUT_SIZE-scaledWidth)/2;
        int top = (INPUT_SIZE-scaledHeight)/2;
        Bitmap scaled = Bitmap.createScaledBitmap(source, scaledWidth, scaledHeight, true);
        Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG | Paint.DITHER_FLAG);
        canvas.drawBitmap(scaled, left, top, paint);
        return output;
    }
    private void updateRunAvailability() { runButton.setEnabled(!busy && modelImage != null && sessions[selectedModel] != null); }
    private void loadModels() {
        status.setText("LOADING MODELS · OFFLINE");
        worker.execute(() -> {
            try {
                environment = OrtEnvironment.getEnvironment(); sessionOptions = new OrtSession.SessionOptions(); sessionOptions.setIntraOpNumThreads(2);
                for (int i=0;i<2;i++) {
                    File file = new File(getCacheDir(), MODEL_FILES[i]);
                    try (InputStream in = getAssets().open(MODEL_FILES[i]); FileOutputStream out = new FileOutputStream(file)) { byte[] buffer = new byte[16384]; int n; while ((n=in.read(buffer))>0) out.write(buffer,0,n); }
                    sessions[i] = environment.createSession(file.getAbsolutePath(), sessionOptions);
                    validateSession(sessions[i]);
                }
                runOnUiThread(() -> { if (closing) return; status.setText("MODELS READY · OFFLINE"); updateRunAvailability(); });
            } catch (Exception e) { runOnUiThread(() -> { if(closing)return; status.setText("MODEL FILES UNAVAILABLE"); toast("Model loading failed. Check ONNX files."); updateRunAvailability(); }); }
        });
    }
    private void validateSession(OrtSession session) {
        if (session.getInputNames().size()!=1 || !session.getInputNames().contains("images") || !session.getOutputNames().contains("probabilities") || !session.getOutputNames().contains("heatmaps")) throw new IllegalStateException("Model must expose images, probabilities, and heatmaps");
    }
    private void runInference() {
        if (busy || modelImage == null || sessions[selectedModel] == null) return;
        final int model = selectedModel; final long token = generation; final Bitmap image = modelImage;
        busy = true; runButton.setText("Analyzing…"); updateRunAvailability(); status.setText("RUNNING ON DEVICE · OFFLINE");
        worker.execute(() -> {
            try {
                float[] tensorData = preprocess(image); String inputName = sessions[model].getInputNames().iterator().next();
                float[] probabilities; float[][][] heat; long elapsed;
                try (OnnxTensor tensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(tensorData), new long[]{1,3,INPUT_SIZE,INPUT_SIZE})) {
                    long started = System.nanoTime();
                    try (OrtSession.Result result = sessions[model].run(Collections.singletonMap(inputName,tensor))) {
                        elapsed = TimeUnit.NANOSECONDS.toMillis(System.nanoTime()-started);
                        probabilities = readProbabilities(result, sessions[model]);
                        heat = readHeatmap(result, sessions[model]);
                    }
                }
                final float[][][] finalHeat=heat; final float[] finalP=probabilities; final long finalElapsed=elapsed;
                runOnUiThread(() -> {
                    if (closing || token != generation || model != selectedModel) { busy=false; runButton.setText("Run inference  →"); updateRunAvailability(); return; }
                    busy=false; runButton.setText("Run inference  →"); render(finalP,finalHeat,finalElapsed,model); status.setText("DONE · ON-DEVICE ONLY"); updateRunAvailability();
                });
            } catch (Exception e) { postFailure(token,e); }
        });
    }
    private void postFailure(long token, Exception e) { runOnUiThread(() -> { if(closing)return; busy=false;runButton.setText("Run inference  →");if(token==generation){status.setText("INFERENCE FAILED");toast("Inference failed. Please try another image.");}updateRunAvailability(); }); }
    static float[] preprocess(Bitmap bitmap) {
        Bitmap b = bitmap.getWidth()==INPUT_SIZE && bitmap.getHeight()==INPUT_SIZE ? bitmap : Bitmap.createScaledBitmap(bitmap,INPUT_SIZE,INPUT_SIZE,true);
        int[] pixels = new int[INPUT_SIZE*INPUT_SIZE]; b.getPixels(pixels,0,INPUT_SIZE,0,0,INPUT_SIZE,INPUT_SIZE); int plane=INPUT_SIZE*INPUT_SIZE; float[] data=new float[3*plane];
        for(int i=0;i<pixels.length;i++){data[i]=((pixels[i]>>16)&255)/255f;data[plane+i]=((pixels[i]>>8)&255)/255f;data[plane*2+i]=(pixels[i]&255)/255f;} return data;
    }
    private float[] readProbabilities(OrtSession.Result result, OrtSession session) throws Exception {
        String preferred=findOutput(session,"probabilities");
        if(preferred!=null){float[] p=toFour(result.get(preferred).get().getValue());if(p!=null)return p;}
        for(String name:session.getOutputNames()){if(name.equals(preferred))continue;float[] p=toFour(result.get(name).get().getValue());if(p!=null)return p;}
        throw new IllegalStateException("No four-class probability output");
    }
    private float[] toFour(Object value) {
        if(value instanceof float[]){float[] p=(float[])value;return p.length==4?p:null;}
        if(value instanceof float[][]){float[][] a=(float[][])value;if(a.length==1&&a[0].length==4)return a[0];}
        if(value instanceof Object[]){for(Object item:(Object[])value){float[] p=toFour(item);if(p!=null)return p;}}
        return null;
    }
    private float[][][] readHeatmap(OrtSession.Result result, OrtSession session) throws Exception {
        String preferred=findOutput(session,"heatmaps");
        if(preferred!=null){float[][][] map=toHeat(result.get(preferred).get().getValue());if(map!=null)return map;}
        for(String name:session.getOutputNames()){if(name.equals(preferred))continue;float[][][] map=toHeat(result.get(name).get().getValue());if(map!=null)return map;}
        return null;
    }
    private float[][][] toHeat(Object value) {
        if(!(value instanceof float[][][][]))return null;
        float[][][][] all=(float[][][][])value;
        if(all.length==0||all[0].length!=4||all[0][0].length==0||all[0][0][0].length==0)return null;
        return all[0];
    }
    private String findOutput(OrtSession session, String fragment) { for(String name:session.getOutputNames())if(name.toLowerCase(Locale.US).contains(fragment))return name;return null; }
    void render(float[] p, float[][][] heat, long ms, int model) {
        if(p.length!=4)return; int best=0; for(int i=1;i<4;i++)if(p[i]>p[best])best=i;
        selectedProbabilities=p.clone(); selectedHeatmap=heat==null?null:makeHeatOverlay(modelImage,heat[best]);
        prediction.setText(CLASSES[best]); confidence.setText(String.format(Locale.US,"%.1f%% model confidence",p[best]*100)); resultRows.removeAllViews();
        for(int i=0;i<4;i++){
            LinearLayout r=row(); TextView name=label(CLASSES[i],13,i==best?WHITE:MUTED,i==best);r.addView(name,new LinearLayout.LayoutParams(dp(82),-2));
            ProgressBar bar=new ProgressBar(this,null,android.R.attr.progressBarStyleHorizontal);bar.setMax(1000);bar.setProgress((int)(Math.max(0,Math.min(1,p[i]))*1000));bar.setProgressTintList(android.content.res.ColorStateList.valueOf(i==best?Color.WHITE:Color.rgb(100,100,100)));bar.setProgressBackgroundTintList(android.content.res.ColorStateList.valueOf(Color.rgb(39,39,39)));r.addView(bar,new LinearLayout.LayoutParams(0,dp(4),1));
            TextView value=label(String.format(Locale.US,"%2.0f%%",p[i]*100),11,MUTED,false);value.setGravity(Gravity.END);r.addView(value,new LinearLayout.LayoutParams(dp(42),-2));LinearLayout.LayoutParams rowp=new LinearLayout.LayoutParams(-1,dp(28));rowp.topMargin=dp(9);resultRows.addView(r,rowp);
        }
        runCount[model]++;averageMs[model]+= (ms-averageMs[model])/runCount[model];
        latency.setText("Inference · "+ms+" ms  ·  avg "+String.format(Locale.US,"%.0f",averageMs[model])+" ms\n"+deviceName()+" · Android API "+Build.VERSION.SDK_INT+" · CPU, 2 threads");
        updateHeatmapButton(); if(showHeatmap&&selectedHeatmap!=null){preview.setImageBitmap(selectedHeatmap);heatmapLegend.setVisibility(View.VISIBLE);} else {preview.setImageBitmap(modelImage);heatmapLegend.setVisibility(View.GONE);}
    }
    static float[] preprocessPairImage(Bitmap image) { return preprocess(makeModelImage(image)); }

    private Bitmap makeHeatOverlay(Bitmap image, float[][] heat) {
        if (image == null || heat == null || heat.length == 0 || heat[0].length == 0) return null;
        int width=image.getWidth(), height=image.getHeight(), rows=heat.length, cols=heat[0].length;
        Bitmap overlay=image.copy(Bitmap.Config.ARGB_8888,true);
        int[] pixels=new int[width*height]; overlay.getPixels(pixels,0,width,0,0,width,height);
        for (int y=0;y<height;y++) {
            float fy=Math.max(0f,Math.min(rows-1f,((y+.5f)*rows/height)-.5f));
            int y0=(int)fy, y1=Math.min(y0+1,rows-1); float wy=fy-y0;
            for (int x=0;x<width;x++) {
                float fx=Math.max(0f,Math.min(cols-1f,((x+.5f)*cols/width)-.5f));
                int x0=(int)fx, x1=Math.min(x0+1,cols-1); float wx=fx-x0;
                float top=heat[y0][x0]*(1-wx)+heat[y0][x1]*wx;
                float bottom=heat[y1][x0]*(1-wx)+heat[y1][x1]*wx;
                float activation=Math.max(0f,Math.min(1f,top*(1-wy)+bottom*wy));
                int heatColor=attentionColor(activation), i=y*width+x, base=pixels[i];
                int red=Math.round(((base>>16)&255)*.55f+Color.red(heatColor)*.45f);
                int green=Math.round(((base>>8)&255)*.55f+Color.green(heatColor)*.45f);
                int blue=Math.round((base&255)*.55f+Color.blue(heatColor)*.45f);
                pixels[i]=Color.rgb(red,green,blue);
            }
        }
        overlay.setPixels(pixels,0,width,0,0,width,height); return overlay;
    }
    private int attentionColor(float value) {
        int[] stops={Color.rgb(45,105,220),Color.rgb(35,190,200),Color.rgb(248,210,65),Color.rgb(235,75,55)};
        float scaled=value*(stops.length-1); int index=Math.min(stops.length-2,(int)scaled); float t=scaled-index;
        int a=stops[index], b=stops[index+1];
        return Color.rgb(Math.round(Color.red(a)+(Color.red(b)-Color.red(a))*t),Math.round(Color.green(a)+(Color.green(b)-Color.green(a))*t),Math.round(Color.blue(a)+(Color.blue(b)-Color.blue(a))*t));
    }
    private String deviceName(){String manufacturer=Build.MANUFACTURER==null?"Android":Build.MANUFACTURER;String model=Build.MODEL==null?"device":Build.MODEL;return (manufacturer+" "+model).trim();}
    private void toggleHeatmap(){if(selectedHeatmap==null)return;showHeatmap=!showHeatmap;preview.setImageBitmap(showHeatmap?selectedHeatmap:modelImage);heatmapLegend.setVisibility(showHeatmap?View.VISIBLE:View.GONE);updateHeatmapButton();}
    private void updateHeatmapButton(){boolean available=selectedHeatmap!=null;heatmapButton.setEnabled(available);heatmapButton.setText(showHeatmap?"Hide heatmap":"Show heatmap");}
    private void clearPrediction(){selectedHeatmap=null;selectedProbabilities=null;showHeatmap=false;prediction.setText("");confidence.setText("");resultRows.removeAllViews();latency.setText("Latency · —");heatmapLegend.setVisibility(View.GONE);updateHeatmapButton();if(modelImage!=null)preview.setImageBitmap(modelImage);}
    @Override protected void onDestroy(){if(artAugView!=null)artAugView.onTabHidden();closing=true;generation++;worker.execute(()->{for(OrtSession session:sessions)if(session!=null)try{session.close();}catch(Exception ignored){}if(sessionOptions!=null)try{sessionOptions.close();}catch(Exception ignored){}});worker.shutdown();super.onDestroy();}
    private void toast(String text){Toast.makeText(this,text,Toast.LENGTH_SHORT).show();}
}




































