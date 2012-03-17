uniform samplerCube skyboxTex;

varying vec3 pos;

void main() {
    vec3 dir = normalize(pos);

	vec3 color = textureCube(skyboxTex, dir).rgb;
	gl_FragColor = vec4(color,1);
}
