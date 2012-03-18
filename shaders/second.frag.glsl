uniform sampler2D phongTex;
uniform sampler2D bloomTex;
uniform sampler2D motionTex;
uniform sampler2D cocTex;
uniform sampler2D shadowTex;
uniform float debug;
varying vec2 texcoord;

void main() {
	vec3 color;
	if (debug == 1.0f) {
	    color = texture2D(phongTex, texcoord);
	} else if (debug == 2.0f) {
	    color = texture2D(bloomTex, texcoord);
    } else if (debug == 3.0f) {
        color = texture2D(motionTex, texcoord);
    } else if (debug == 4.0f) {
        color = texture2D(cocTex, texcoord);
    } else if (debug == 5.0f) {
        color = vec3(texture2D(shadowTex, texcoord).z);
    }

	gl_FragColor = vec4(color,1);
}
