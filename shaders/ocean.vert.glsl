attribute vec3 positionIn;
attribute vec3 partialUIn;
attribute vec3 partialVIn;
attribute float foamTimeIn;
attribute float foamAlphaIn;

uniform mat4 ModelViewMatrix;
uniform mat4 ProjectionMatrix;
uniform mat4 LightViewMatrix;
uniform mat4 LightProjectionMatrix;
uniform mat3 NormalMatrix;

varying vec3 eyePosition;
varying vec3 normal;
varying vec4 shadowcoord;
varying vec3 lightDir;
varying vec2 texcoord;
varying float foamTime;
varying float foamAlpha;

void main() {
    texcoord = positionIn.xz / 100.0;
    shadowcoord = 0.5 *(LightProjectionMatrix * LightViewMatrix * vec4(positionIn, 1)) + vec4(0.5,0.5,0.5,0.5);;
	vec4 eyeTemp =  ModelViewMatrix * vec4(positionIn, 1);

	//vec3 crossNormal = cross(partialUIn, -partialVIn);
	vec3 crossNormal = cross(partialUIn, -partialVIn);
	normal = NormalMatrix * crossNormal;
	//normal = crossNormal;

	lightDir = (NormalMatrix * vec3(1.0, 0.5, 1.0)).xyz;

	eyePosition = eyeTemp.xyz;
	gl_Position = ProjectionMatrix * eyeTemp;
	foamTime = foamTimeIn;
	foamAlpha = foamAlphaIn;
}
