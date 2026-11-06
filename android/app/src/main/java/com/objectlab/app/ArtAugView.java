package com.objectlab.app;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Typeface;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.exifinterface.media.ExifInterface;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Executor;

/** Small offline gallery for approved, pre-generated ArtAug photo pairs. */
final class ArtAugView extends LinearLayout {
    private static final int WHITE = Color.WHITE;
    private static final int MUTED = Color.rgb(160, 160, 160);
    private static final int PANEL = Color.rgb(20, 20, 20);
    private static final String[] CLASS_ORDER = {"bottle", "cup", "pen", "scissors"};

    private final Context context;
    private final Executor worker;
    private final List<Example> examples = new ArrayList<>();
    private final int[] classPositions = {-1, -1, -1, -1};
    private final ImageView originalImage, enhancedImage;
    private final TextView empty, classTitle, originalCaption, enhancedCaption;
    private final Button generateButton;
    private int selected = -1;
    private int classCursor = 0;
    private long generation = 0;

    ArtAugView(Context context, Executor worker) {
        super(context);
        this.context = context;
        this.worker = worker;
        setOrientation(VERTICAL);
        setVisibility(GONE);

        TextView intro = text("ArtAug enhancements were generated offline.", 12, MUTED, false);
        intro.setLineSpacing(dp(2), 1);
        addView(intro);

        generateButton = new Button(context);
        generateButton.setText("Generate");
        generateButton.setTextSize(14);
        generateButton.setAllCaps(false);
        generateButton.setTypeface(null, Typeface.BOLD);
        generateButton.setTextColor(Color.BLACK);
        generateButton.setMinHeight(dp(52));
        generateButton.setBackground(background(Color.rgb(245, 245, 245), 13));
        generateButton.setContentDescription("Show the next saved offline ArtAug photo");
        generateButton.setOnClickListener(v -> generateNext());
        LayoutParams actionLp = new LayoutParams(-1, dp(52));
        actionLp.topMargin = dp(8);
        addView(generateButton, actionLp);

        classTitle = text("", 16, WHITE, true);
        LayoutParams classLp = new LayoutParams(-1, -2);
        classLp.topMargin = dp(14);
        addView(classTitle, classLp);

        originalImage = image("Original photo");
        LayoutParams originalLp = new LayoutParams(-1, dp(225));
        originalLp.topMargin = dp(8);
        addView(originalImage, originalLp);
        originalCaption = caption("ORIGINAL", MUTED);
        addView(originalCaption);

        enhancedImage = image("ArtAug photo enhanced by AI");
        LayoutParams enhancedLp = new LayoutParams(-1, dp(225));
        enhancedLp.topMargin = dp(8);
        addView(enhancedImage, enhancedLp);
        enhancedCaption = caption("ENHANCED BY AI", WHITE);
        addView(enhancedCaption);

        empty = text("Tap Generate to view the first saved pair.", 13, MUTED, true);
        empty.setGravity(Gravity.CENTER);
        empty.setPadding(dp(14), dp(25), dp(14), dp(25));
        empty.setBackground(background(PANEL, 14));
        LayoutParams emptyLp = new LayoutParams(-1, -2);
        emptyLp.topMargin = dp(12);
        addView(empty, emptyLp);

        originalImage.setVisibility(GONE);
        enhancedImage.setVisibility(GONE);
        originalCaption.setVisibility(GONE);
        enhancedCaption.setVisibility(GONE);
        classTitle.setVisibility(GONE);
        loadManifest();
    }

    int getSelectedIndex() { return selected; }
    int getClassCursor() { return classCursor; }

    void onTabHidden() { generation++; }
    void onTabShown() { if (selected >= 0) show(selected); }

    void setSelectionState(int index, int cursor) {
        classCursor = Math.floorMod(cursor, CLASS_ORDER.length);
        if (!examples.isEmpty() && index >= 0 && index < examples.size()) {
            selected = index;
            show(index);
        }
    }

    private void generateNext() {
        if (examples.isEmpty()) {
            empty.setText("No saved ArtAug pairs are installed yet. Import reviewed offline pairs, then tap Generate.");
            return;
        }
        int classIndex = classCursor % CLASS_ORDER.length;
        classCursor = (classCursor + 1) % CLASS_ORDER.length;
        String wanted = CLASS_ORDER[classIndex];
        int start = classPositions[classIndex] < 0 ? 0 : (classPositions[classIndex] + 1) % examples.size();
        int found = -1;
        for (int offset = 0; offset < examples.size(); offset++) {
            int candidate = (start + offset) % examples.size();
            if (sameClass(examples.get(candidate).className, wanted)) {
                found = candidate;
                break;
            }
        }
        if (found < 0) {
            classTitle.setVisibility(View.VISIBLE);
            classTitle.setText(titleClass(wanted) + " · NO SAVED PHOTO");
            empty.setVisibility(View.VISIBLE);
            empty.setText("No saved photo for this class. Import a reviewed offline pair to continue.");
            hidePair();
            return;
        }
        classPositions[classIndex] = found;
        selected = found;
        show(found);
    }

    private void loadManifest() {
        try (InputStream in = context.getAssets().open("artaug/manifest.json")) {
            byte[] bytes = readBounded(in, 1_500_000);
            JSONObject root = new JSONObject(new String(bytes, java.nio.charset.StandardCharsets.UTF_8));
            JSONArray entries = root.optJSONArray("examples");
            if (entries != null) {
                for (int i = 0; i < entries.length(); i++) {
                    JSONObject item = entries.optJSONObject(i);
                    if (item == null || !item.optBoolean("approved", false)) continue;
                    Example example = Example.from(item);
                    if (example != null) examples.add(example);
                }
            }
            if (examples.isEmpty()) {
                generateButton.setEnabled(false);
                generateButton.setAlpha(.55f);
                empty.setText("No saved ArtAug pairs are installed yet. Import reviewed offline pairs to begin.");
            }
        } catch (Exception ignored) {
            generateButton.setEnabled(false);
            generateButton.setAlpha(.55f);
            empty.setText("No saved ArtAug pairs are installed yet. Import reviewed offline pairs to begin.");
        }
    }

    private void show(int index) {
        if (examples.isEmpty()) return;
        selected = Math.floorMod(index, examples.size());
        Example entry = examples.get(selected);
        long token = ++generation;
        generateButton.setEnabled(false);
        classTitle.setVisibility(VISIBLE);
        classTitle.setText(titleClass(entry.className) + " · " + (selected + 1) + "/" + examples.size());
        empty.setVisibility(VISIBLE);
        empty.setText("Loading saved pair…");
        hidePair();
        originalImage.setImageDrawable(null);
        enhancedImage.setImageDrawable(null);
        worker.execute(() -> {
            Bitmap original = null, enhanced = null;
            try {
                original = decode(entry.original);
                enhanced = decode(entry.enhanced);
            } catch (Exception ignored) { }
            final Bitmap source = original, edited = enhanced;
            post(() -> {
                if (token != generation) return;
                generateButton.setEnabled(true);
                if (source == null || edited == null) {
                    empty.setText("This saved pair could not be opened. Tap Generate to try another.");
                    return;
                }
                empty.setVisibility(GONE);
                showPair();
                originalImage.setImageBitmap(source);
                enhancedImage.setImageBitmap(edited);
            });
        });
    }

    private void hidePair() {
        originalImage.setVisibility(GONE);
        enhancedImage.setVisibility(GONE);
        originalCaption.setVisibility(GONE);
        enhancedCaption.setVisibility(GONE);
    }

    private void showPair() {
        originalImage.setVisibility(VISIBLE);
        enhancedImage.setVisibility(VISIBLE);
        originalCaption.setVisibility(VISIBLE);
        enhancedCaption.setVisibility(VISIBLE);
    }

    private Bitmap decode(String relative) throws Exception {
        String path = "artaug/" + relative;
        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        try (InputStream in = context.getAssets().open(path)) { BitmapFactory.decodeStream(in, null, bounds); }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) throw new IllegalArgumentException("Invalid image");
        int max = Math.max(bounds.outWidth, bounds.outHeight), sample = 1;
        while (max / sample > 1500) sample *= 2;
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = sample;
        Bitmap image;
        try (InputStream in = context.getAssets().open(path)) { image = BitmapFactory.decodeStream(in, null, options); }
        if (image == null) throw new IllegalArgumentException("Invalid image");
        int orientation;
        try (InputStream in = context.getAssets().open(path)) {
            orientation = new ExifInterface(in).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
        }
        return MainActivity.applyExifOrientation(image, orientation);
    }

    private boolean sameClass(String actual, String wanted) {
        String normalized = actual.toLowerCase(Locale.ROOT).replaceAll("[^a-z]", "");
        if (wanted.equals("bottle")) return normalized.equals("bottle") || normalized.equals("bottles");
        if (wanted.equals("cup")) return normalized.equals("cup") || normalized.equals("cups");
        if (wanted.equals("pen")) return normalized.equals("pen") || normalized.equals("pens");
        return normalized.equals("scissor") || normalized.equals("scissors");
    }

    private String titleClass(String value) {
        if (sameClass(value, "bottle")) return "Bottles";
        if (sameClass(value, "cup")) return "Cups";
        if (sameClass(value, "pen")) return "Pens";
        if (sameClass(value, "scissors")) return "Scissors";
        String normalized = value.toLowerCase(Locale.ROOT);
        return normalized.isEmpty() ? value : normalized.substring(0, 1).toUpperCase(Locale.ROOT) + normalized.substring(1);
    }

    private ImageView image(String description) {
        ImageView image = new ImageView(context);
        image.setScaleType(ImageView.ScaleType.FIT_CENTER);
        image.setBackground(background(PANEL, 14));
        image.setClipToOutline(true);
        image.setContentDescription(description);
        return image;
    }
    private TextView caption(String text, int color) {
        TextView label = text(text, 10, color, true);
        label.setGravity(Gravity.CENTER_VERTICAL);
        return label;
    }
    private TextView text(String value, int size, int color, boolean bold) {
        TextView text = new TextView(context);
        text.setText(value); text.setTextSize(size); text.setTextColor(color);
        if (bold) text.setTypeface(null, Typeface.BOLD);
        return text;
    }
    private android.graphics.drawable.GradientDrawable background(int color, int radius) {
        android.graphics.drawable.GradientDrawable drawable = new android.graphics.drawable.GradientDrawable();
        drawable.setColor(color); drawable.setCornerRadius(dp(radius));
        drawable.setStroke(dp(1), Color.rgb(45, 45, 45));
        return drawable;
    }
    private int dp(float value) { return (int)(value * getResources().getDisplayMetrics().density + .5f); }
    private static byte[] readBounded(InputStream in, int limit) throws Exception {
        java.io.ByteArrayOutputStream output = new java.io.ByteArrayOutputStream();
        byte[] buffer = new byte[8192]; int count, total = 0;
        while ((count = in.read(buffer)) > 0) {
            total += count;
            if (total > limit) throw new IllegalArgumentException("Manifest too large");
            output.write(buffer, 0, count);
        }
        return output.toByteArray();
    }

    private static final class Example {
        final String id, className, original, enhanced;
        Example(String id, String className, String original, String enhanced) {
            this.id = id; this.className = className; this.original = original; this.enhanced = enhanced;
        }
        static Example from(JSONObject item) {
            String id = item.optString("id", "").trim();
            String className = item.optString("class_name", "").trim();
            String original = safePath(item.optString("original", ""));
            String enhanced = safePath(item.optString("enhanced", ""));
            if (id.isEmpty() || className.isEmpty() || original == null || enhanced == null) return null;
            return new Example(id, className, original, enhanced);
        }
        static String safePath(String path) {
            path = path.replace('\\', '/');
            if (path.isEmpty() || path.startsWith("/") || path.contains(":") || path.contains("..")) return null;
            return path;
        }
    }
}
