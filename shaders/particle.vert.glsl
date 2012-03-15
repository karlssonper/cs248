attribute vec3 positionIn;
attribute vec3 colorIn;
attribute float timeIn;
attribute float sizeIn;

uniform mat4 ModelViewMatrix;
uniform mat4 ProjectionMatrix;

varying vec3 eyeSpacePos;
varying vec3 color;
varying float time;

void main() {

	vec4 eyeTemp = (ModelViewMatrix*vec4(positionIn, 1.0));
	eyeSpacePos = eyeTemp.xyz;
	gl_PointSize = sizeIn / length(eyeTemp);
	gl_Position = ProjectionMatrix * eyeTemp;
	color = colorIn;
	time = timeIn;

}
