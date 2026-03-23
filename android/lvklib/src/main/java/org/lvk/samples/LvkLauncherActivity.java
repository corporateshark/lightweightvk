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
  private boolean waitingForPermission = false;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    if (!Environment.isExternalStorageManager()) {
      waitingForPermission = true;
      Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);
      intent.setData(Uri.fromParts("package", getPackageName(), null));
      startActivity(intent);
    } else {
      launchNativeActivity();
    }
  }

  @Override
  protected void onRestart() {
    super.onRestart();
    if (waitingForPermission) {
      waitingForPermission = false;
      launchNativeActivity();
    }
  }

  private void launchNativeActivity() {
    Intent intent = new Intent();
    intent.setClassName(this, "org.lvk.samples.MainActivity");
    startActivity(intent);
    finish();
  }
}
