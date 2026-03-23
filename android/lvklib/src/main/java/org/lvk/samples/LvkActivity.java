package org.lvk.samples;

import android.os.Bundle;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.view.WindowManager;

public class LvkActivity extends android.app.NativeActivity {

  private void hideSystemBars() {
    final WindowInsetsController controller = getWindow().getInsetsController();
    if (controller != null) {
      // Set the behavior first so the hide policy is in place before bars are dismissed
      controller.setSystemBarsBehavior(WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);
      controller.hide(WindowInsets.Type.systemBars());
    }
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    getWindow().setDecorFitsSystemWindows(false);
    getWindow().getAttributes().layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_ALWAYS;

    super.onCreate(savedInstanceState);

    // Defer the initial hide: the InsetsController is not fully ready during onCreate
    // (the "skip insets animation" log indicates the hide is ignored if called too early)
    getWindow().getDecorView().post(this::hideSystemBars);

    // Re-hide system bars whenever they become visible (e.g., after surface recreation).
    // Guard with isVisible() and defer via post() to avoid a recursive loop
    // (controller.hide() dispatches new insets which would re-trigger this listener).
    getWindow().getDecorView().setOnApplyWindowInsetsListener((view, insets) -> {
      if (insets.isVisible(WindowInsets.Type.statusBars()) || insets.isVisible(WindowInsets.Type.navigationBars())) {
        view.post(this::hideSystemBars);
      }
      return view.onApplyWindowInsets(insets);
    });
  }

  @Override
  public void onAttachedToWindow() {
    super.onAttachedToWindow();
    hideSystemBars();
  }

  @Override
  protected void onResume() {
    super.onResume();
    hideSystemBars();
  }

  @Override
  public void onWindowFocusChanged(boolean hasFocus) {
    super.onWindowFocusChanged(hasFocus);
    if (hasFocus) {
      hideSystemBars();
    }
  }

  @SuppressWarnings("deprecation")
  @Override
  public void onBackPressed() {
    System.gc();
    System.exit(0);
  }
}
