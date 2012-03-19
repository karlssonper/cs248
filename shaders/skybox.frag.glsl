uniform samplerCube skyboxTex;

varying vec3 pos;
varying float coc;
varying float eyeDepth;

float rgb2lum(vec3 color)
{
    return 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;

}

vec3 bloom(vec3 color, float lumTresh)
{
    return color *
           (1.0 / (1.0 - lumTresh)) *
           clamp(rgb2lum(color) - lumTresh, 0.0, 1.0);
}

void main() {
    vec3 dir = normalize(pos);

	vec3 color = textureCube(skyboxTex, dir).rgb;
	float alpha = clamp(eyeDepth/200,0,1);
	gl_FragData[0] = vec4(color,1.0);

	gl_FragData[1] = vec4(bloom(color,0.7),1);

	//gl_FragData[2] = vec4(coc,eyeDepth,0.0,1.0);
	gl_FragData[2] = vec4(coc, alpha, 1.0, 1.0);

	gl_FragData[3] = vec4(1.0, 1.0, 1.0, 1.0);
}
