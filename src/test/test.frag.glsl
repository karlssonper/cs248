varying vec3 color;
varying float time;

void main() {	
	if (time < 0.0) discard;
	gl_FragColor = vec4(color, max(time, 0.0));
}