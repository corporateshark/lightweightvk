/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if LVK_WITH_SDL3

#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

void* createCocoaWindowView(void* window, void** outLayer) {
  NSWindow* nswindow = (NSWindow*)window;
  CAMetalLayer* layer = [CAMetalLayer layer];
  layer.device = MTLCreateSystemDefaultDevice();
  layer.opaque = YES;
  layer.displaySyncEnabled = YES;
  NSScreen* screen = [NSScreen mainScreen];
  CGFloat factor = [screen backingScaleFactor];
  layer.contentsScale = factor;
  nswindow.contentView.layer = layer;
  nswindow.contentView.wantsLayer = YES;

  *outLayer = layer;

  return nswindow.contentView;
}
#endif // LVK_WITH_SDL3
