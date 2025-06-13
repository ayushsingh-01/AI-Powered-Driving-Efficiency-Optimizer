import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk, scrolledtext

def simulate_data(n=1000):
    np.random.seed(42)
    speed = np.random.uniform(20, 120, n)
    terrain = np.random.choice([0, 1], n)
    braking = np.random.choice([0, 1], n)
    energy = 0.18*speed + 4.5*terrain + 2.5*braking + np.random.normal(0, 2, n)
    return pd.DataFrame({'speed': speed, 'terrain': terrain, 'braking': braking, 'energy': energy})

def terrain_str(val):
    return 'Hilly' if val == 1 else 'Flat'

def braking_str(val):
    return 'Harsh' if val == 1 else 'Gentle'

def train_model(data):
    X = data[['speed', 'terrain', 'braking']]
    y = data['energy']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score

def suggest_tips(speed, terrain, braking):
    tips = []
    if speed > 100:
        tips.append('Reduce speed below 100 km/h to improve efficiency.')
    if terrain == 1:
        tips.append('Maintain steady speed on hilly terrain.')
    if braking == 1:
        tips.append('Brake gently to save energy.')
    if not tips:
        tips.append('Your driving is already efficient!')
    return tips

def show_results(samples, model, data, r2_score):
    def display_sample(idx):
        st.config(state=tk.NORMAL)
        st.delete(1.0, tk.END)
        if idx < 0 or idx >= len(data):
            st.insert(tk.END, 'Invalid index. Please enter a value between 0 and {}.\n'.format(len(data)-1), 'error')
        else:
            row = data.iloc[idx]
            sample = pd.DataFrame({'speed': [row["speed"]], 'terrain': [row["terrain"]], 'braking': [row["braking"]]})
            pred = model.predict(sample)[0]
            tips = suggest_tips(row["speed"], row["terrain"], row["braking"])
            st.insert(tk.END, f'Input: Speed={row["speed"]:.1f} km/h, Terrain={terrain_str(row["terrain"])} ({int(row["terrain"])}), Braking={braking_str(row["braking"])} ({int(row["braking"])}))\n', 'input')
            st.insert(tk.END, f'Predicted energy usage: {pred:.2f} kWh\n', 'prediction')
            st.insert(tk.END, 'Driving tips:\n', 'tips')
            for tip in tips:
                st.insert(tk.END, f'- {tip}\n', 'tipitem')
            st.insert(tk.END, '\n' + '-'*90 + '\n', 'divider')
        st.config(state=tk.DISABLED)

    def display_custom(speed, terrain, braking):
        st.config(state=tk.NORMAL)
        st.delete(1.0, tk.END)
        try:
            speed = float(speed)
            terrain = int(terrain)
            braking = int(braking)
            if terrain not in [0,1] or braking not in [0,1]:
                raise ValueError
            sample = pd.DataFrame({'speed': [speed], 'terrain': [terrain], 'braking': [braking]})
            pred = model.predict(sample)[0]
            tips = suggest_tips(speed, terrain, braking)
            st.insert(tk.END, f'Input: Speed={speed:.1f} km/h, Terrain={terrain_str(terrain)} ({terrain}), Braking={braking_str(braking)} ({braking}))\n', 'input')
            st.insert(tk.END, f'Predicted energy usage: {pred:.2f} kWh\n', 'prediction')
            st.insert(tk.END, 'Driving tips:\n', 'tips')
            for tip in tips:
                st.insert(tk.END, f'- {tip}\n', 'tipitem')
        except:
            st.insert(tk.END, 'Please enter valid values: Speed (number 20 - 120), Terrain (0 or 1), Braking (0 or 1).\n', 'error')
        st.config(state=tk.DISABLED)

    root = tk.Tk()
    root.title('EV Driving Efficiency Optimizer')
    root.geometry('850x750')
    root.configure(bg='#f0f4f8')

    header = tk.Label(root, text='EV Driving Efficiency Optimizer', font=('Segoe UI', 22, 'bold'), bg='#2d6cdf', fg='white', pady=15)
    header.pack(fill=tk.X)

    info_panel = tk.Frame(root, bg='#f0f4f8')
    info_panel.pack(padx=20, pady=(10, 0), fill=tk.X)
    r2_label = tk.Label(info_panel, text=f'Model R² Score: {r2_score:.3f}', font=('Segoe UI', 13, 'bold'), fg='#0a7d2c', bg='#f0f4f8')
    r2_label.pack(anchor='w')
    steps_text = (
        'AI Workflow Steps:\n'
        '1. Simulate 1000 driving scenarios (speed, terrain, braking, energy usage).\n'
        '2. Train a Random Forest model to predict energy usage from driving behavior.\n'
        '3. Evaluate model performance (R² score above).\n'
        '4. Display 10 random test results with predictions and tips.\n'
        '5. Enter any test index or your own details to view predictions and tips.'
    )
    steps_label = tk.Label(info_panel, text=steps_text, font=('Segoe UI', 11), fg='#222', bg='#f0f4f8', justify='left')
    steps_label.pack(anchor='w', pady=(5, 0))

    panel = tk.Frame(root, bg='white', bd=2, relief=tk.RIDGE)
    panel.pack(padx=20, pady=15, fill=tk.BOTH, expand=True)

    st = scrolledtext.ScrolledText(panel, width=110, height=22, font=('Consolas', 11), bg='#f9fafb', fg='#222', bd=0, wrap=tk.WORD)
    st.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    st.tag_config('input', foreground='#2d6cdf', font=('Consolas', 11, 'bold'))
    st.tag_config('prediction', foreground='#0a7d2c', font=('Consolas', 11, 'bold'))
    st.tag_config('tips', foreground='#b36a00', font=('Consolas', 11, 'bold'))
    st.tag_config('tipitem', foreground='#b36a00', font=('Consolas', 11))
    st.tag_config('divider', foreground='#888', font=('Consolas', 11, 'bold'))
    st.tag_config('error', foreground='#c00', font=('Consolas', 11, 'bold'))

    st.config(state=tk.NORMAL)
    st.insert(tk.END, 'Showing 10 random test results:\n\n', 'input')
    for idx, row in samples.iterrows():
        sample = pd.DataFrame({'speed': [row["speed"]], 'terrain': [row["terrain"]], 'braking': [row["braking"]]})
        pred = model.predict(sample)[0]
        tips = suggest_tips(row["speed"], row["terrain"], row["braking"])
        st.insert(tk.END, f'Index: {row.name}\n', 'input')
        st.insert(tk.END, f'Input: Speed={row["speed"]:.1f} km/h, Terrain={terrain_str(row["terrain"])} ({int(row["terrain"])}), Braking={braking_str(row["braking"])} ({int(row["braking"])}))\n', 'input')
        st.insert(tk.END, f'Predicted energy usage: {pred:.2f} kWh\n', 'prediction')
        st.insert(tk.END, 'Driving tips:\n', 'tips')
        for tip in tips:
            st.insert(tk.END, f'- {tip}\n', 'tipitem')
        st.insert(tk.END, '\n' + '-'*90 + '\n', 'divider')
    st.config(state=tk.DISABLED)

    controls = tk.Frame(root, bg='#f0f4f8')
    controls.pack(pady=(0, 10))
    tk.Label(controls, text='Enter test index (0-{}):'.format(len(data)-1), font=('Segoe UI', 11), bg='#f0f4f8').pack(side=tk.LEFT, padx=(0, 5))
    idx_entry = tk.Entry(controls, width=8, font=('Segoe UI', 11))
    idx_entry.pack(side=tk.LEFT, padx=5)
    def on_show():
        try:
            idx = int(idx_entry.get())
            display_sample(idx)
        except:
            st.config(state=tk.NORMAL)
            st.delete(1.0, tk.END)
            st.insert(tk.END, 'Please enter a valid integer index.\n', 'error')
            st.config(state=tk.DISABLED)
    show_btn = tk.Button(controls, text='Show Result', font=('Segoe UI', 11, 'bold'), bg='#2d6cdf', fg='white', activebackground='#174a8c', activeforeground='white', command=on_show)
    show_btn.pack(side=tk.LEFT, padx=10)

    user_panel = tk.Frame(root, bg='#f0f4f8')
    user_panel.pack(pady=(0, 10))
    tk.Label(user_panel, text='Or enter your own details:', font=('Segoe UI', 11, 'bold'), bg='#f0f4f8').grid(row=0, column=0, padx=5, pady=2)
    tk.Label(user_panel, text='Speed (km/h) (20 to 120):', font=('Segoe UI', 11), bg='#f0f4f8').grid(row=0, column=1, padx=2)
    speed_entry = tk.Entry(user_panel, width=8, font=('Segoe UI', 11))
    speed_entry.grid(row=0, column=2, padx=2)
    tk.Label(user_panel, text='Terrain (0=Flat, 1=Hilly):', font=('Segoe UI', 11), bg='#f0f4f8').grid(row=0, column=3, padx=2)
    terrain_entry = tk.Entry(user_panel, width=4, font=('Segoe UI', 11))
    terrain_entry.grid(row=0, column=4, padx=2)
    tk.Label(user_panel, text='Braking (0=Gentle, 1=Harsh):', font=('Segoe UI', 11), bg='#f0f4f8').grid(row=0, column=5, padx=2)
    braking_entry = tk.Entry(user_panel, width=4, font=('Segoe UI', 11))
    braking_entry.grid(row=0, column=6, padx=2)
    def on_custom():
        display_custom(speed_entry.get(), terrain_entry.get(), braking_entry.get())
    custom_btn = tk.Button(user_panel, text='Get My Tips', font=('Segoe UI', 11, 'bold'), bg='#0a7d2c', fg='white', activebackground='#0a7d2c', activeforeground='white', command=on_custom)
    custom_btn.grid(row=0, column=7, padx=10)

    root.mainloop()

def main():
    data = simulate_data()
    model, score = train_model(data)
    samples = data.sample(10, random_state=1)
    show_results(samples, model, data, score)

if __name__ == '__main__':
    main()
