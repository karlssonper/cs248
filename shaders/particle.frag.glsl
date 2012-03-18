uniform sampler2D sprite;

varying float timeCopy;
varying vec2 texCoord;

void main() {	
	if (timeCopy < 0.0) discard;
	gl_FragColor = texture2D(sprite, texCoord) * vec4(1.0, 1.0, 1.0, timeCopy);
	//gl_FragColor = vec4(1.0, 1.0, 1.0, time);
	//gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
