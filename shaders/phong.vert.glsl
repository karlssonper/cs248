attribute vec3 positionIn;

uniform mat4 ModelViewMatrix;
uniform mat4 ProjectionMatrix;
//uniform mat4 NormalMatrix;

void main() {

	// Transform the vertex to get the eye-space position of the vertex
	/*vec4 eyeTemp =  ModelViewMatrix * vec4(positionIn, 1);
	eyePosition = eyeTemp.xyz;*/
	
	// Transform again to get the clip-space position.  The gl_Position
	// variable tells OpenGL where the vertex should go.
	gl_Position = ProjectionMatrix * ModelViewMatrix *vec4(positionIn, 1);

	// Transform the normal, just like in Assignment 2.
	
    /*normal = (NormalMatrix * vec4(normalIn,0)).xyz;
	tangent =   (NormalMatrix * vec4(tangentIn,0)).xyz;
	bitangent =   (NormalMatrix* vec4(bitangentIn,0)).xyz;

	L = (NormalMatrix * vec4(0,-1,0,0)).xyz;

	// Just copy the texture coordinates
	texcoord = texcoordIn;*/
}
