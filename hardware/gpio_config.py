"""
GPIO Pin Configuration for Trading System Hardware Interface

Pin assignments for Raspberry Pi 5 GPIO header.
All GPIO numbers are BCM numbering.
"""

# GPIO Chip for Pi 5
GPIO_CHIP = '/dev/gpiochip0'

# I2C Configuration (Screen)
I2C_BUS = 1
LCD_ADDR = 0x27    # 20x4 I2C LCD display
LCD_COLS = 20
LCD_ROWS = 4

# LCD Orientation (set True if display is mounted upside down)
LCD_FLIPPED = False

# LED Configuration (Common Anode - LOW = ON, HIGH = OFF)
# LED 1: System Health
LED1_RED = 23    # Physical pin 16
LED1_GREEN = 24  # Physical pin 18
LED1_BLUE = 25   # Physical pin 22

# LED 2: Trading Status
LED2_RED = 26    # Physical pin 37
LED2_GREEN = 16  # Physical pin 36
LED2_BLUE = 20   # Physical pin 38

# LED 3: Research Engine
LED3_RED = 19    # Physical pin 35
LED3_GREEN = 21  # Physical pin 40
LED3_BLUE = 12   # Physical pin 32

# All LED pins grouped
LED_PINS = {
    'system': {'red': LED1_RED, 'green': LED1_GREEN, 'blue': LED1_BLUE},
    'trading': {'red': LED2_RED, 'green': LED2_GREEN, 'blue': LED2_BLUE},
    'research': {'red': LED3_RED, 'green': LED3_GREEN, 'blue': LED3_BLUE},
}

ALL_LED_PINS = [
    LED1_RED, LED1_GREEN, LED1_BLUE,
    LED2_RED, LED2_GREEN, LED2_BLUE,
    LED3_RED, LED3_GREEN, LED3_BLUE,
]

# Encoder Configuration
# Encoder 1: Trading Screen Control
ENC1_CLK = 22    # Physical pin 15 (S1)
ENC1_DT = 17     # Physical pin 11 (S2)
ENC1_SW = 27     # Physical pin 13 (Key/button)

# Encoder 2: Research Screen Control
ENC2_CLK = 6     # Physical pin 31 (was labeled DT)
ENC2_DT = 13     # Physical pin 33 (was labeled SW)
ENC2_SW = 5      # Physical pin 29 (was labeled CLK)

ENCODER_PINS = {
    'trading': {'clk': ENC1_CLK, 'dt': ENC1_DT, 'sw': ENC1_SW},
    'research': {'clk': ENC2_CLK, 'dt': ENC2_DT, 'sw': ENC2_SW},
}

ALL_ENCODER_PINS = [ENC1_CLK, ENC1_DT, ENC1_SW, ENC2_CLK, ENC2_DT, ENC2_SW]

# Color Definitions (which pins to turn ON for each color)
# For common anode LEDs, pins in list are set LOW (on), others HIGH (off)
COLORS = {
    'off': [],
    'red': ['red'],
    'green': ['green'],
    'blue': ['blue'],
    'yellow': ['red', 'green'],
    'cyan': ['green', 'blue'],
    'magenta': ['red', 'blue'],
    'white': ['red', 'green', 'blue'],
}

# Status color mappings
STATUS_COLORS = {
    # System health
    'healthy': 'green',
    'warning': 'yellow',
    'error': 'red',
    'offline': 'off',

    # Trading status
    'active': 'green',
    'idle': 'blue',
    'stopped': 'off',
    'pending': 'yellow',

    # Research status
    'evolving': 'blue',
    'complete': 'green',
    'failed': 'red',
    'paused': 'yellow',
}
