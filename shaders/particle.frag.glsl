uniform sampler2D sprite;
uniform sampler2D cocTex;

varying float timeCopy;
varying vec2 texCoord;
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

float saturate(float bgDepth, float fragDepth, float scale) {
	float diff = bgDepth - fragDepth;
	if (diff > scale) {
		return 1.0f;
	} else {
		return diff/scale;
	}
}


void main() {	
	if (timeCopy < 0.0) discard;
	vec4 color = texture2D(sprite, texCoord);

	float worldU = gl_FragCoord.x / 800.0;
	float worldV = gl_FragCoord.y / 600.0;

	float bgDepth = texture2D(cocTex, vec2(worldU, worldV)).y;
	float fragDepth = clamp(-eyeDepth/200,0,1);

	float alpha;
	if (bgDepth < fragDepth) {
		alpha = 0.0;
	} else {
		alpha = saturate(bgDepth, fragDepth, 0.1);
	}

	//skriv farg till denna
	gl_FragData[0] = color * vec4(1.0, 1.0, 1.0, alpha*timeCopy);

	//gl_FragData[1] = vec4(bloom(color.rgb,0.7),1);
	//gl_FragColor = vec4(1.0, 1.0, 1.0, time);
	//gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
