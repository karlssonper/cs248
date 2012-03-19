attribute vec3 positionIn;
attribute vec2 texcoordIn;
attribute vec3 normalIn;
attribute vec3 tangentIn;
attribute vec3 bitangentIn;

uniform mat4 ModelViewMatrix;
uniform mat4 ProjectionMatrix;
uniform mat4 ModelMatrix;
uniform mat4 LightViewMatrix;
uniform mat4 LightProjectionMatrix;
uniform mat3 NormalMatrix;

uniform float focalPlane;
uniform float nearBlurPlane;
uniform float farBlurPlane;
uniform float maxBlur;

varying vec2 texcoord;
varying vec3 eyePosition;
varying vec3 normal;
varying vec4 shadowcoord;
varying float coc;
//varying vec3 tangent;
//varying vec3 bitangent;
varying vec3 L;

float calculateCoC(float depth)
{
    float f;
    if (depth < focalPlane) {
        f = (depth - focalPlane)/(focalPlane - nearBlurPlane);
    }
    else
    {
        f = (depth - focalPlane)/(farBlurPlane - focalPlane);
    // clamp the far blur to a maximum blurriness
        f = clamp (f, 0, maxBlur);
    }
    // scale and bias into [0, 1] range
    return f * 0.5 + 0.5;

}

void main() {
	vec4 eyeTemp =  ModelViewMatrix * vec4(positionIn, 1);
	eyePosition = eyeTemp.xyz;
	coc = calculateCoC(-eyePosition.z);

	vec4 worldPos = ModelMatrix * vec4(positionIn, 1);
	shadowcoord = 0.5 *(LightProjectionMatrix * LightViewMatrix * worldPos) +
                    vec4(0.5,0.5,0.5,0.5);

	gl_Position = ProjectionMatrix * eyeTemp;

    normal = NormalMatrix * normalIn;
	//tangent =   NormalMatrix * tangentIn;
	//bitangent =   NormalMatrix * bitangentIn;


	L = (NormalMatrix * vec3(1.0, 0.5, 1.0)).xyz;

	texcoord = texcoordIn;
}
