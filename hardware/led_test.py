#!/usr/bin/env python3
"""LED test script - cycles all LEDs through R/G/B using existing library"""

import sys
import time
sys.path.insert(0, '/home/thiselazar/trading_system')

from hardware.leds import LEDController

if __name__ == '__main__':
    print("LED Test - Cycling R/G/B on all LEDs")
    print("Press Ctrl+C to stop")

    leds = LEDController()
    colors = ['red', 'green', 'blue']
    led_names = ['system', 'trading', 'research']

    try:
        while True:
            for color in colors:
                print(f"  All LEDs: {color.upper()}")
                for name in led_names:
                    leds.set_color(name, color)
                time.sleep(0.7)
    except KeyboardInterrupt:
        print("\nStopping...")
        for name in led_names:
            leds.set_color(name, 'off')
        print("Done")
