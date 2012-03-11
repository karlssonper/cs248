attribute vec3 positionIn;

uniform mat4 ModelViewMatrix;
uniform mat4 ProjectionMatrix;

varying vec3 eyePosition;

void main() {
	vec4 eyeTemp =  ModelViewMatrix * vec4(positionIn, 1);
	eyePosition = eyeTemp.xyz;
	gl_Position = ProjectionMatrix * eyeTemp;
}
