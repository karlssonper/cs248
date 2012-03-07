
// These are the "input" to our shader.  They are read from the vertex
// arrays that we specified in the C++ code.
attribute vec3 positionIn;
attribute vec2 texcoordIn;
attribute vec3 normalIn;
attribute vec3 tangentIn;
attribute vec3 bitangentIn;

uniform float hasNormalTex;
uniform mat4 modelMatrix;

// These are the "output" values of our shader.  OpenGL will interpolate
// these for us (good riddance project 2!)
varying vec2 texcoord;
varying vec3 normal;
varying vec3 eyePosition;
varying vec3 tangent;
varying vec3 bitangent;
varying vec4 shadowcoord;

void main() {

	// Transform the vertex to get the eye-space position of the vertex
	vec4 eyeTemp =  gl_ModelViewMatrix * vec4(positionIn, 1);
	eyePosition = eyeTemp.xyz;
	
	vec4 worldPos = modelMatrix * vec4(positionIn, 1);
	shadowcoord = 0.5 *(gl_TextureMatrix[6] * gl_TextureMatrix[5] * worldPos) + vec4(0.5,0.5,0.5,0.5);

	// Transform again to get the clip-space position.  The gl_Position
	// variable tells OpenGL where the vertex should go.
	gl_Position = gl_ProjectionMatrix * eyeTemp;

	// Transform the normal, just like in Assignment 2.
	
    normal = gl_NormalMatrix * normalIn;
	tangent =   gl_NormalMatrix *tangentIn;
	bitangent =   gl_NormalMatrix* bitangentIn;

	// Just copy the texture coordinates
	texcoord = texcoordIn;
}
