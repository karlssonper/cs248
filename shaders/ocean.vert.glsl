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

uniform float focalPlane;
uniform float nearBlurPlane;
uniform float farBlurPlane;
uniform float maxBlur;

varying vec3 eyePosition;
varying vec3 normal;
varying vec4 shadowcoord;
varying vec3 lightDir;
varying vec2 texcoord;
varying float foamTime;
varying float foamAlpha;
varying float coc;

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
    //return f;
    return f*f*f*f;
}

void main() {
    texcoord.x = positionIn.x/50.0;
    texcoord.y = positionIn.z/100.0;
    shadowcoord = 0.5 *(LightProjectionMatrix * LightViewMatrix * vec4(positionIn, 1)) + vec4(0.5,0.5,0.5,0.5);;
	vec4 eyeTemp =  ModelViewMatrix * vec4(positionIn, 1);

	//vec3 crossNormal = cross(partialUIn, -partialVIn);
	vec3 crossNormal = cross(partialUIn, -partialVIn);
	normal = NormalMatrix * crossNormal;
	//normal = crossNormal;

	lightDir = (NormalMatrix * vec3(1.0, 0.5, 1.0)).xyz;

	eyePosition = eyeTemp.xyz;
	coc = calculateCoC(-eyePosition.z);
	gl_Position = ProjectionMatrix * eyeTemp;
	foamTime = foamTimeIn;
	foamAlpha = foamAlphaIn;
}
