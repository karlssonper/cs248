attribute vec3 positionIn;

uniform mat4 ModelViewMatrix;
uniform mat4 ProjectionMatrix;

varying vec3 pos;

void main() {
    pos = positionIn;
	gl_Position = ProjectionMatrix * ModelViewMatrix * vec4(positionIn,1);
}
