import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import os
import pandas as pd
import datetime
from src.collector import HandDataCollector

import threading
import time

class HandCollectorApp(ctk.CTk):
    """Modern GUI for hand landmark data collection."""
    
    def __init__(self):
        super().__init__()
        
        self.title("Hand Landmark Data Collector - AI Pipeline")
        self.geometry("1100x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Data storage
        self.data_storage = []
        self.is_recording = False
        self.current_label = "default"
        self.save_path = "data/"
        os.makedirs(self.save_path, exist_ok=True)
        
        # Landmark collector
        self.collector = HandDataCollector()
        
        # --- UI Layout ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Sidebar for controls
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="ML Collector", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.pack(pady=20, padx=20)
        
        # Label Input
        self.label_input = ctk.CTkEntry(self.sidebar, placeholder_text="Data Category (e.g. Wave, Fist)")
        self.label_input.pack(pady=10, padx=20, fill="x")
        
        # Stats Display
        self.stats_label = ctk.CTkLabel(self.sidebar, text="Samples: 0", font=ctk.CTkFont(size=14))
        self.stats_label.pack(pady=10, padx=20)
        
        # Action Buttons
        self.record_btn = ctk.CTkButton(self.sidebar, text="Start Recording", fg_color="green", hover_color="#006400", command=self.toggle_recording)
        self.record_btn.pack(pady=10, padx=20)
        
        self.save_btn = ctk.CTkButton(self.sidebar, text="Save Data (CSV)", command=self.save_to_csv)
        self.save_btn.pack(pady=10, padx=20)
        
        self.clear_btn = ctk.CTkButton(self.sidebar, text="Clear Current", fg_color="red", command=self.clear_data)
        self.clear_btn.pack(pady=10, padx=20)
        
        # Video Display Frame
        self.video_frame = ctk.CTkFrame(self, corner_radius=10)
        self.video_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True, fill="both")
        
        # Camera feed
        self.cap = cv2.VideoCapture(0)
        self.update_video()
        
    def toggle_recording(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.current_label = self.label_input.get() or "no_label"
            self.record_btn.configure(text="Stop Recording", fg_color="red")
        else:
            self.record_btn.configure(text="Start Recording", fg_color="green")
            
    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # Flip frame for mirror view
            frame = cv2.flip(frame, 1)
            
            # Process with Mediapipe
            annotated_frame, hand_data_list = self.collector.process_frame(frame)
            
            # Collect data if recording
            if self.is_recording and hand_data_list:
                for hand in hand_data_list:
                    # Capture timestamp and metadata
                    row = {
                        'timestamp': datetime.datetime.now().isoformat(),
                        'label': self.current_label,
                        'handedness': hand['label'],
                        'score': hand['score']
                    }
                    # Flatten landmarks: x0, y0, z0, x1, y1, z1...
                    for idx, lm in enumerate(hand['landmarks']):
                        row[f"x{idx}"] = lm['x']
                        row[f"y{idx}"] = lm['y']
                        row[f"z{idx}"] = lm['z']
                        
                    self.data_storage.append(row)
                    self.stats_label.configure(text=f"Samples: {len(self.data_storage)}")
            
            # Update GUI
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            img_tk = ImageTk.PhotoImage(image=img)
            self.video_label.img_tk = img_tk
            self.video_label.configure(image=img_tk)
            
        self.after(10, self.update_video)
        
    def save_to_csv(self):
        if not self.data_storage:
            print("No data to save.")
            return
        
        try:
            df = pd.DataFrame(self.data_storage)
            filename = f"hand_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = os.path.join(self.save_path, filename)
            df.to_csv(filepath, index=False)
            print(f"Data saved to {filepath}")
            # Reset after save
            self.data_storage = []
            self.stats_label.configure(text="Samples: 0 (Saved!)")
        except Exception as e:
            print(f"Error saving data: {e}")
            
    def clear_data(self):
        self.data_storage = []
        self.stats_label.configure(text="Samples: 0")

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    app = HandCollectorApp()
    app.mainloop()
