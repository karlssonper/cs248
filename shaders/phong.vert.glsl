attribute vec3 positionIn;
attribute vec2 texcoordIn;
attribute vec3 normalIn;
attribute vec3 tangentIn;
attribute vec3 bitangentIn;

uniform mat4 ModelViewMatrix;
uniform mat4 ProjectionMatrix;
uniform mat3 NormalMatrix;

varying vec2 texcoord;
varying vec3 eyePosition;
varying vec3 normal;
//varying vec3 tangent;
//varying vec3 bitangent;
varying vec3 L;

void main() {
	vec4 eyeTemp =  ModelViewMatrix * vec4(positionIn, 1);
	eyePosition = eyeTemp.xyz;

	gl_Position = ProjectionMatrix * eyeTemp;

    normal = NormalMatrix * normalIn;
	//tangent =   NormalMatrix * tangentIn;
	//bitangent =   NormalMatrix * bitangentIn;

	L = NormalMatrix * vec3(-1,1,1);

	texcoord = texcoordIn;
}