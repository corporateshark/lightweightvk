package org.lvk.samples;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.Settings;

/**
 * Trampoline activity that requests storage permissions before launching the NativeActivity.
 * This avoids starting the native thread before permissions are granted.
 */
public class LvkLauncherActivity extends Activity {
  private static final int REQUEST_MANAGE_STORAGE_CODE = 1;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    if (!Environment.isExternalStorageManager()) {
      Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);
      intent.setData(Uri.fromParts("package", getPackageName(), null));
      startActivityForResult(intent, REQUEST_MANAGE_STORAGE_CODE);
    } else {
      launchNativeActivity();
    }
  }

  private void launchNativeActivity() {
    Intent intent = new Intent();
    intent.setClassName(this, "org.lvk.samples.MainActivity");
    startActivity(intent);
    finish();
  }

  @SuppressWarnings("deprecation")
  @Override
  protected void onActivityResult(int requestCode, int resultCode, Intent data) {
    super.onActivityResult(requestCode, resultCode, data);
    if (requestCode == REQUEST_MANAGE_STORAGE_CODE) {
      launchNativeActivity();
    }
  }
}
