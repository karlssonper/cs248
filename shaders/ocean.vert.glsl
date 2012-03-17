attribute vec3 positionIn;
attribute vec3 partialUIn;
attribute vec3 partialVIn;
attribute float foldIn;

uniform mat4 ModelViewMatrix;
uniform mat4 ProjectionMatrix;
uniform mat3 NormalMatrix;

varying vec3 eyePosition;
varying vec3 normal;
varying vec3 lightDir;

void main() {
	vec4 eyeTemp =  ModelViewMatrix * vec4(positionIn, 1);


	//normal  = NormalMatrix *cross( vec3(0.0, slopeIn.y*0.5, 2.0 / 128.0), vec3(2.0 / 128.0, slopeIn.x*0.5, 0.0));

	vec3 crossNormal = cross(partialUIn,partialVIn);
	//normal = NormalMatrix * crossNormal;
	normal = crossNormal;
	//normal = partialUIn;
	lightDir = NormalMatrix * vec3(0.0, -1.0, 0.0);

	eyePosition = eyeTemp.xyz;
	gl_Position = ProjectionMatrix * eyeTemp;
}
