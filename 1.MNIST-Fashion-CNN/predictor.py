import tensorflow as tf
import numpy as np
import json
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw, ImageTk

class FashionMNISTPredictor:
    def __init__(self, model_path='fashion_mnist_model.h5', class_names_path='class_names.json'):
        """Initialize the predictor with trained model and class names"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
            print(f"Model loaded successfully from {model_path}")
            print(f"Classes: {self.class_names}")
        except Exception as e:
            raise Exception(f"Failed to load model or class names: {e}")
    
    def preprocess_image(self, pil_image):
        """Preprocess PIL image for prediction"""
        # Resize to 28x28
        img = pil_image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Reshape for model input
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    
    def predict(self, pil_image, top_k=3):
        """Make prediction on PIL image"""
        # Preprocess image
        img_array = self.preprocess_image(pil_image)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        probabilities = tf.nn.softmax(predictions[0]).numpy()
        
        # Get top k predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            class_name = self.class_names[idx]
            confidence = probabilities[idx] * 100
            results.append({
                'class': class_name,
                'confidence': confidence,
                'index': idx
            })
        
        return results

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fashion MNIST Drawing Predictor")
        self.root.geometry("700x600")
        
        # Canvas settings
        self.canvas_size = 280  # 10x the original 28x28
        self.brush_size = 15
        
        # Initialize predictor
        try:
            self.predictor = FashionMNISTPredictor()
        except Exception as e:
            messagebox.showerror("Error", f"Could not load model: {e}")
            return
        
        self.setup_ui()
        self.reset_canvas()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Instructions
        instructions = ttk.Label(main_frame, 
                               text="Draw a fashion item (shirt, pants, shoe, etc.) and click 'Predict'",
                               font=("Arial", 12))
        instructions.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Canvas frame
        canvas_frame = ttk.LabelFrame(main_frame, text="Drawing Canvas", padding="10")
        canvas_frame.grid(row=1, column=0, columnspan=3, pady=(0, 10))
        
        # Canvas
        self.canvas = tk.Canvas(canvas_frame, 
                               width=self.canvas_size, 
                               height=self.canvas_size, 
                               bg='black', 
                               cursor='pencil')
        self.canvas.pack()
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)
        
        # Controls frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=2, column=0, columnspan=3, pady=(0, 10))
        
        # Brush size control
        ttk.Label(controls_frame, text="Brush Size:").pack(side=tk.LEFT, padx=(0, 5))
        self.brush_var = tk.IntVar(value=self.brush_size)
        brush_scale = ttk.Scale(controls_frame, from_=5, to=30, 
                               variable=self.brush_var, orient=tk.HORIZONTAL, length=150)
        brush_scale.pack(side=tk.LEFT, padx=(0, 20))
        brush_scale.bind('<Motion>', self.update_brush_size)
        
        # Buttons
        ttk.Button(controls_frame, text="Clear", command=self.reset_canvas).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Predict", command=self.predict).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Save Drawing", command=self.save_drawing).pack(side=tk.LEFT, padx=5)
        
        # Prediction results
        self.result_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="10")
        self.result_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.result_text = tk.Text(self.result_frame, height=8, width=60, font=("Courier", 11))
        scrollbar = ttk.Scrollbar(self.result_frame, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def reset_canvas(self):
        self.canvas.delete("all")
        # Create PIL image for drawing
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 0)  # Grayscale, black background
        self.draw_image = ImageDraw.Draw(self.image)
        self.drawing = False
        
        # Clear results
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, "Draw something and click 'Predict' to see results...")
        
    def update_brush_size(self, event=None):
        self.brush_size = self.brush_var.get()
        
    def start_draw(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
        
    def draw(self, event):
        if self.drawing:
            # Draw on canvas
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, 
                                  width=self.brush_size, fill='white', capstyle=tk.ROUND, smooth=tk.TRUE)
            
            # Draw on PIL image
            self.draw_image.line([self.last_x, self.last_y, event.x, event.y], 
                               fill=255, width=self.brush_size)
            
            self.last_x = event.x
            self.last_y = event.y
            
    def stop_draw(self, event):
        self.drawing = False
        
    def predict(self):
        try:
            # Make prediction
            results = self.predictor.predict(self.image, top_k=5)
            
            # Display results
            result_text = "🎯 PREDICTION RESULTS\n"
            result_text += "=" * 40 + "\n\n"
            
            for i, result in enumerate(results):
                confidence = result['confidence']
                class_name = result['class']
                
                # Add emoji based on confidence
                if confidence > 50:
                    emoji = "🎯"
                elif confidence > 30:
                    emoji = "👍"
                elif confidence > 15:
                    emoji = "🤔"
                else:
                    emoji = "❓"
                
                result_text += f"{emoji} #{i+1}: {class_name}\n"
                result_text += f"    Confidence: {confidence:.1f}%\n"
                result_text += f"    {'█' * int(confidence/5)} {confidence:.1f}%\n\n"
            
            # Add interpretation
            best_prediction = results[0]
            if best_prediction['confidence'] > 70:
                interpretation = f"🎉 I'm very confident this is a {best_prediction['class']}!"
            elif best_prediction['confidence'] > 40:
                interpretation = f"🤖 I think this is probably a {best_prediction['class']}"
            elif best_prediction['confidence'] > 20:
                interpretation = f"🤷 My best guess is {best_prediction['class']}, but I'm not very sure"
            else:
                interpretation = "😅 I'm having trouble recognizing this drawing"
            
            result_text += "💭 INTERPRETATION:\n"
            result_text += interpretation + "\n\n"
            result_text += "💡 TIP: Try drawing clearer shapes for better results!"
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(1.0, result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")
    
    def save_drawing(self):
        try:
            # Save the current drawing
            filename = f"drawing_{len(list(self.image.getdata()))}.png"
            self.image.save(filename)
            messagebox.showinfo("Saved", f"Drawing saved as {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

def main():
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()