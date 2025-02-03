Reconstruir la imagen y ejecutar un nuevo contenedor:

Paso 1: Reconstruir la imagen:
docker build -t mi-app-streamlit .
Paso 2: Ejecutar un nuevo contenedor:
docker run -p 8501:8501 mi-app-streamlit

Uso de volúmenes (opcional, para desarrollo):
Si deseas ver los cambios sin reconstruir la imagen cada vez, puedes montar el directorio de tu código local dentro del contenedor utilizando un volumen. Por ejemplo, en el comando docker run:

docker run -p 8501:8501 -v $(pwd):/app mi-app-streamlit
Esto asume que tu código está en el directorio actual y que en tu Dockerfile el directorio de trabajo es /app. Con esto, los cambios en el código se reflejarán en el contenedor sin tener que reconstruir la imagen.
Nota: Esta opción es ideal para desarrollo, pero no se recomienda en producción.

En resumen, para subir los cambios a un contenedor ya detenido, la opción más común es reconstruir la imagen y luego ejecutar un nuevo contenedor.






