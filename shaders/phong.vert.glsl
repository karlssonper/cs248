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
varying vec3 tangent;
varying vec3 bitangent;
varying vec3 L;

void main() {

	// Transform the vertex to get the eye-space position of the vertex
	vec4 eyeTemp =  ModelViewMatrix * vec4(positionIn, 1);
	eyePosition = eyeTemp.xyz;
	
	// Transform again to get the clip-space position.  The gl_Position
	// variable tells OpenGL where the vertex should go.
	gl_Position = ProjectionMatrix * ModelViewMatrix *vec4(positionIn, 1);

	// Transform the normal, just like in Assignment 2.
	
    normal = NormalMatrix * normalIn;
	tangent =   NormalMatrix * tangentIn;
	bitangent =   NormalMatrix * bitangentIn;

	L = NormalMatrix * vec3(0,-1,0);

	// Just copy the texture coordinates
	texcoord = texcoordIn;
}
