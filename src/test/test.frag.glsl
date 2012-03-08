uniform sampler2D sprite;

varying vec3 color;
varying float time;

void main() {	
	if (time < 0.0) discard;
	gl_FragColor = texture2D(sprite, gl_PointCoord.xy) * vec4(1.0, 1.0, 1.0, time);
	//gl_FragColor = vec4(1.0, 1.0, 1.0, time);
}