import trimesh
import numpy as np

# Приклад створення об'єкта (куба)
vertices = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
])
faces = np.array([
    [0, 1, 2], [1, 3, 2],
    # ... інші грані
])

# Проблема може бути тут: неправильні розміри або типи
try:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    print("Об'єкт Trimesh створено успішно!")
except Exception as e:
    # Помилка 'name' може бути викликана невірним форматом даних.
    print(f"[ПОМИЛКА TRIMESH] Не вдалося створити об'єкт: {e}")