uniform sampler2D sprite;

varying float timeCopy;
varying vec2 texCoord;

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
	if (timeCopy < 0.0) discard;
	vec4 color = texture2D(sprite, texCoord);
	gl_FragData[1] = color * vec4(1.0, 1.0, 1.0, timeCopy);

	gl_FragData[2] = vec4(bloom(color.rgb,0.7),1);
	//gl_FragColor = vec4(1.0, 1.0, 1.0, time);
	//gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
