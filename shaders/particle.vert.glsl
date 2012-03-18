#version 330

attribute vec3 positionIn;
attribute vec3 colorIn;
attribute float timeIn;
attribute float sizeIn;

uniform mat4 ModelViewMatrix;

varying float size;
varying float time;

void main() {
    gl_Position = ModelViewMatrix*vec4(positionIn, 1.0);
    time = timeIn;
    size = 0.001*sizeIn;
}
