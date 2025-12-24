import cv2
import mediapipe as mp
import numpy as np
import math
import time

try:
    import screen_brightness_control as sbc
    BRIGHTNESS_AVAILABLE = True
except ImportError:
    BRIGHTNESS_AVAILABLE = False
    print("⚠️  Install 'screen-brightness-control' for brightness control:")
    print("   pip install screen-brightness-control")

# ----------------- BRIGHTNESS CONTROL -----------------
def set_brightness(level):
    """Set screen brightness (0-100)"""
    if BRIGHTNESS_AVAILABLE:
        try:
            level = max(0, min(100, int(level*2)))
            sbc.set_brightness(level)
        except Exception as e:
            print(f"Brightness error: {e}")

# ----------------- MEDIAPIPE -----------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def draw_particle_ring(frame, center, radius, time_val, color, num_particles=24):
    """Draw animated particle ring"""
    for i in range(num_particles):
        angle = (2 * math.pi / num_particles) * i + time_val * 2
        x = int(center[0] + math.cos(angle) * radius)
        y = int(center[1] + math.sin(angle) * radius)
        
        # Particle with glow
        cv2.circle(frame, (x, y), 4, color, -1)
        cv2.circle(frame, (x, y), 6, color, 1)

def draw_mandala_pattern(frame, center, radius, time_val, color):
    """Draw sacred geometry mandala pattern"""
    num_points = 12
    for i in range(num_points):
        angle1 = (2 * math.pi / num_points) * i + time_val * 0.5
        angle2 = (2 * math.pi / num_points) * (i + 2) + time_val * 0.5
        
        # Outer connection points
        x1 = int(center[0] + math.cos(angle1) * radius * 0.85)
        y1 = int(center[1] + math.sin(angle1) * radius * 0.85)
        x2 = int(center[0] + math.cos(angle2) * radius * 0.85)
        y2 = int(center[1] + math.sin(angle2) * radius * 0.85)
        
        # Draw connecting lines with gradient effect
        cv2.line(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

def draw_energy_spiral(frame, center, radius, time_val, color):
    """Draw energy spiral emanating from center"""
    num_spirals = 3
    points_per_spiral = 50
    
    for spiral_idx in range(num_spirals):
        pts = []
        offset = (2 * math.pi / num_spirals) * spiral_idx
        
        for i in range(points_per_spiral):
            t = i / points_per_spiral
            angle = t * 4 * math.pi + time_val * 2 + offset
            r = radius * t * 0.7
            
            x = int(center[0] + math.cos(angle) * r)
            y = int(center[1] + math.sin(angle) * r)
            pts.append([x, y])
        
        pts = np.array(pts, np.int32)
        cv2.polylines(frame, [pts], False, color, 1, cv2.LINE_AA)

def draw_mystic_shield(frame, center, radius, time_val, brightness_level):
    """Draw the complete mystic shield with enhanced effects"""
    cx, cy = center
    
    # Create overlay for transparency effects
    overlay = frame.copy()
    
    # Pulsating effect
    pulse = int(12 * math.sin(time_val * 3))
    active_radius = radius + pulse
    
    # ----------------- OUTER GLOW RINGS -----------------
    # Multiple layers for depth
    for i, r_offset in enumerate([35, 28, 21, 14, 7]):
        r = active_radius + r_offset
        alpha = 0.02 + (i * 0.005)
        
        # Gradient from cyan to golden
        t_norm = i / 5
        color = (
            int(0 + t_norm * 255),      # Blue to golden
            int(140 + t_norm * 115),    # 
            int(255 - t_norm * 55)      # Cyan to gold
        )
        
        cv2.circle(overlay, (cx, cy), r, color, 3, cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # ----------------- MAIN SHIELD RINGS -----------------
    # Outer bright ring (cyan-gold gradient)
    cv2.circle(frame, (cx, cy), active_radius + 8, (0, 200, 255), 4, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), active_radius, (0, 255, 255), 3, cv2.LINE_AA)
    
    # Inner golden ring
    cv2.circle(frame, (cx, cy), int(active_radius * 0.75), (50, 220, 255), 2, cv2.LINE_AA)
    
    # ----------------- SACRED GEOMETRY -----------------
    # Draw mandala pattern
    draw_mandala_pattern(frame, (cx, cy), active_radius * 0.9, time_val, (0, 200, 255))
    
    # Inner hexagram
    hex_points = []
    for i in range(6):
        angle = (2 * math.pi / 6) * i + time_val * 0.3
        x = int(cx + math.cos(angle) * radius * 0.5)
        y = int(cy + math.sin(angle) * radius * 0.5)
        hex_points.append([x, y])
    
    hex_points = np.array(hex_points, np.int32)
    cv2.polylines(frame, [hex_points], True, (100, 255, 255), 2, cv2.LINE_AA)
    
    # ----------------- ENERGY PARTICLES -----------------
    # Rotating particle rings at different speeds
    draw_particle_ring(frame, (cx, cy), active_radius * 0.65, time_val, (255, 220, 100), 16)
    draw_particle_ring(frame, (cx, cy), active_radius * 0.85, -time_val * 0.7, (100, 255, 255), 20)
    
    # ----------------- ENERGY SPIRALS -----------------
    draw_energy_spiral(frame, (cx, cy), active_radius, time_val, (0, 180, 255))
    
    # ----------------- RADIAL ENERGY BEAMS -----------------
    num_beams = 32
    for i in range(num_beams):
        if i % 2 == 0:  # Alternating pattern
            angle = (2 * math.pi / num_beams) * i + time_val
            
            # Inner point
            x1 = int(cx + math.cos(angle) * radius * 0.15)
            y1 = int(cy + math.sin(angle) * radius * 0.15)
            
            # Outer point with variation
            variation = 0.92 + 0.08 * math.sin(time_val * 3 + i)
            x2 = int(cx + math.cos(angle) * radius * variation)
            y2 = int(cy + math.sin(angle) * radius * variation)
            
            # Color gradient based on angle
            color_shift = int(50 * math.sin(time_val + i))
            color = (max(0, color_shift), 180 + color_shift, 255)
            
            cv2.line(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    
    # ----------------- CENTER CORE -----------------
    # Pulsating center core
    core_pulse = int(8 * math.sin(time_val * 4))
    core_radius = 15 + core_pulse
    
    # Outer glow
    cv2.circle(overlay, (cx, cy), core_radius + 10, (0, 255, 255), -1)
    frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)
    
    # Core layers
    cv2.circle(frame, (cx, cy), core_radius, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), core_radius - 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), core_radius - 8, (100, 220, 255), -1, cv2.LINE_AA)
    
    return frame

def draw_activation_prompt(frame, left, right, distance_val):
    """Draw visual prompt when hands are close for activation"""
    cx = (left[0] + right[0]) // 2
    cy = (left[1] + right[1]) // 2
    
    # Pulsating circle
    pulse_radius = int(40 + 10 * math.sin(time.time() * 5))
    
    # Draw pulsating circle
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), pulse_radius, (0, 255, 100), 2, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), pulse_radius - 10, (0, 200, 100), 1, cv2.LINE_AA)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    
    # Draw text prompt
    text = "BRING CLOSER TO TOGGLE"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
    text_x = cx - text_size[0] // 2
    text_y = cy - 60
    
    # Background for text
    cv2.rectangle(frame, 
                  (text_x - 10, text_y - 25), 
                  (text_x + text_size[0] + 10, text_y + 10),
                  (0, 0, 0), -1)
    
    cv2.putText(frame, text, (text_x, text_y), font, 0.6, (0, 255, 100), 2)
    
    # Distance indicator
    distance_text = f"{int(distance_val)}px"
    cv2.putText(frame, distance_text, (cx - 30, cy), font, 0.7, (0, 255, 100), 2)
    
    return frame

# ----------------- MAIN LOOP -----------------
cap = cv2.VideoCapture(0)
start_time = time.time()

print("🔮 Dr. Strange Mystic Shield - Toggle Mode")
print("━" * 50)
print("Controls:")
print("  • Bring hands CLOSE (< 30px) to TOGGLE shield ON/OFF")
print("  • When shield is ON:")
print("    - Spread hands apart to INCREASE brightness")
print("    - Bring hands together to DECREASE brightness")
print("  • Press 'Q' to quit")
print("━" * 50)

# Shield state
shield_active = False
previous_distance = 0
toggle_cooldown = 0
TOGGLE_COOLDOWN_FRAMES = 30  # Prevent rapid toggling
ACTIVATION_THRESHOLD = 30  # Distance threshold for activation

# Initial brightness
current_brightness = 50
if BRIGHTNESS_AVAILABLE:
    set_brightness(current_brightness)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    
    frame_count += 1
    if toggle_cooldown > 0:
        toggle_cooldown -= 1

    if result.pose_landmarks:
        h, w, _ = frame.shape
        lm = result.pose_landmarks.landmark

        # Get wrist positions
        lw = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
        rw = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]

        left = (int(lw.x * w), int(lw.y * h))
        right = (int(rw.x * w), int(rw.y * h))

        # Calculate distance between wrists
        d = distance(left, right)
        
        # ----------------- TOGGLE MECHANISM -----------------
        # Check if hands crossed the threshold
        if (previous_distance >= ACTIVATION_THRESHOLD and 
            d < ACTIVATION_THRESHOLD and 
            toggle_cooldown == 0):
            
            # Toggle shield state
            shield_active = not shield_active
            toggle_cooldown = TOGGLE_COOLDOWN_FRAMES
            
            # Visual and audio feedback
            if shield_active:
                print(f"✨ Shield ACTIVATED at frame {frame_count}")
            else:
                print(f"💫 Shield DEACTIVATED at frame {frame_count}")
        
        previous_distance = d
        
        # ----------------- SHIELD RENDERING -----------------
        if shield_active:
            # Shield geometry
            cx = (left[0] + right[0]) // 2
            cy = (left[1] + right[1]) // 2
            radius = max(50, min(int(d // 2), 300))

            # ----------------- BRIGHTNESS MAPPING -----------------
            target_brightness = int(np.interp(radius, [50, 300], [10, 100]))
            
            # Smooth brightness transition
            current_brightness += (target_brightness - current_brightness) * 0.1
            current_brightness = max(10, min(100, current_brightness))
            
            set_brightness(int(current_brightness))

            # ----------------- DRAW SHIELD -----------------
            t = time.time() - start_time
            frame = draw_mystic_shield(frame, (cx, cy), radius, t, current_brightness)

            # Draw hand markers with glow
            for hand_pos in [left, right]:
                cv2.circle(frame, hand_pos, 12, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, hand_pos, 8, (255, 255, 255), -1, cv2.LINE_AA)

            # Connection line between hands
            cv2.line(frame, left, right, (0, 200, 255), 2, cv2.LINE_AA)

            # ----------------- HUD INFO -----------------
            info_y = 30
            cv2.putText(frame, f"Shield Radius: {radius}px", 
                        (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 255), 2)
            
            cv2.putText(frame, f"Brightness: {int(current_brightness)}%", 
                        (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 255), 2)
            
            # Brightness bar
            bar_x, bar_y, bar_w, bar_h = 10, info_y + 50, 200, 20
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), 
                         (50, 50, 50), -1)
            
            fill_w = int((current_brightness / 100) * bar_w)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), 
                         (0, 255, 255), -1)
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), 
                         (0, 255, 255), 2)
            
            # Active status indicator
            cv2.putText(frame, "SHIELD: ACTIVE", (w - 220, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
        
        else:
            # Shield is inactive - show hand markers only
            for hand_pos in [left, right]:
                cv2.circle(frame, hand_pos, 10, (100, 100, 100), 2, cv2.LINE_AA)
                cv2.circle(frame, hand_pos, 6, (150, 150, 150), -1, cv2.LINE_AA)

            # Connection line (dimmed)
            cv2.line(frame, left, right, (100, 100, 100), 1, cv2.LINE_AA)
            
            # Show activation prompt when hands are getting close
            if d < ACTIVATION_THRESHOLD + 50:
                frame = draw_activation_prompt(frame, left, right, d)
            
            # Inactive status
            cv2.putText(frame, "SHIELD: INACTIVE", (w - 240, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
            
            # Instructions
            inst_y = h - 60
            cv2.putText(frame, "Bring hands together (< 30px) to activate", 
                        (10, inst_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (150, 150, 150), 1)
            
            # Distance display
            cv2.putText(frame, f"Distance: {int(d)}px", 
                        (10, inst_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (150, 150, 150), 1)

    # Main title
    title_text = "MYSTIC ARTS SHIELD - TOGGLE MODE"
    cv2.putText(frame, title_text, (10, h - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Dr. Strange Mystic Shield", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n✨ Mystic Shield Deactivated")