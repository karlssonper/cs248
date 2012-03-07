attribute vec3 positionIn;
attribute vec3 colorIn;
attribute float timeIn;

varying vec3 eyeSpacePos;
varying vec3 color;
varying float time;


void main() {

	vec4 eyeTemp = (gl_ModelViewMatrix*vec4(positionIn, 1.0));
	eyeSpacePos = eyeTemp.xyz;
	gl_PointSize = 50.0 / length(eyeTemp);
	gl_Position = gl_ProjectionMatrix * eyeTemp;
	color = colorIn;
	time = timeIn;

}
