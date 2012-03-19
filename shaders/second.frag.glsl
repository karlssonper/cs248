uniform sampler2D phongTex;
uniform sampler2D particlesTex;
uniform sampler2D bloomTex;
uniform sampler2D cocTex;
uniform sampler2D shadowTex;
uniform sampler2D depthTex;
uniform sampler2D hudTex;
uniform float debug;
varying vec2 texcoord;

uniform float texDx;

float distance(vec2 tcoords, vec2 uv)
{
    float dx = tcoords.x - uv.x;
    float dy = tcoords.y - uv.y;
    return sqrt(dx*dx + dy*dy);
}

vec3 gaussianBlur(int n)
{
    vec3 sum = vec3(0.0);
    float v = texcoord.y - float(n)/2.0 * texDx;
    float totWeight = 0.0;
    for (int i = 0; i < n; ++i, v += texDx){
        float d = distance(texcoord, vec2(texcoord.x,v ));
        float weight = exp(-d*d);
        sum += weight*texture2D(bloomTex, vec2(texcoord.x,v)).rgb;
        totWeight += weight;
    }
    return sum / totWeight;
}

void main() {
	vec3 color;
	if (debug == 1.0) {
	    vec4 hud = texture2D(hudTex, texcoord);
	    vec4 part = texture2D(particlesTex, texcoord);
	    vec3 phong = texture2D(phongTex, texcoord).rgb;
	    vec3 temp = (1.0-part.a) * phong + part.a*part.rgb;
	    color = (1.0-hud.a)*temp + hud.a*(hud.rgb);
	} else if (debug == 2.0) {
	    color = gaussianBlur(10);
    } else if (debug == 3.0) {
        color = vec3(texture2D(particlesTex, texcoord).a);
        //color = texture2D(phongTex, texcoord) + gaussianBlur(10);
    } else if (debug == 4.0) {
        color = vec3(texture2D(depthTex, texcoord).r/2.0);
    } else if (debug == 5.0) {
        color = vec3(texture2D(shadowTex, texcoord).z);
    }

	gl_FragColor = vec4(color,1);
}
