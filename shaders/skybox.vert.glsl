attribute vec3 positionIn;

uniform mat4 ModelViewMatrix;
uniform mat4 ProjectionMatrix;

uniform float focalPlane;
uniform float nearBlurPlane;
uniform float farBlurPlane;
uniform float maxBlur;

varying vec3 pos;
varying float coc;
varying float eyeDepth;

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
    pos = positionIn;
    vec4 eyePos = ModelViewMatrix * vec4(positionIn,1);
    coc = calculateCoC(-eyePos.z);
    eyeDepth = eyePos.z;
	gl_Position = ProjectionMatrix * eyePos;
}
