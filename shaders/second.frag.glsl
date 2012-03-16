uniform sampler2D phongTex;
uniform sampler2D bloomTex;
uniform sampler2D motionTex;
uniform sampler2D cocTex;

varying vec2 texcoord;

void main() {
	vec3 color = texture2D(phongTex, texcoord).rgb;
	gl_FragColor = vec4(color,1);
}
