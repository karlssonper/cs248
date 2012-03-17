attribute vec3 positionIn;

uniform mat4 ModelViewMatrix;
uniform mat4 ProjectionMatrix;

void main() {
    vec4 eyeTemp = ModelViewMatrix * vec4(positionIn, 1);
    gl_Position = ProjectionMatrix * eyeTemp;
}
