uniform sampler2D phongTex;
uniform sampler2D cocTex;

uniform float texDx;

varying vec2 texcoord;

float distance(vec2 tcoords, vec2 uv)
{
    float dx = tcoords.x - uv.x;
    float dy = tcoords.y - uv.y;
    return sqrt(dx*dx + dy*dy);
}

vec3 gaussianBlur(sampler2D tex, int n)
{
    vec3 sum = vec3(0.0);
    float u = texcoord.x - float(n)/2.0 * texDx;
    float totWeight = 0.0;
    for (int i = 0; i < n; ++i, u += texDx){
        float d = distance(texcoord, vec2(u,texcoord.y ));
        float weight = exp(-d*d);
        sum += weight*texture2D(tex, vec2(u,texcoord.y )).rgb;
        totWeight += weight;
    }
    return sum / totWeight;
}

void main() {
	//First color buffer
	gl_FragColor = vec4(gaussianBlur(phongTex,1),1);
    //gl_FragColor = vec4(1,0,0,1);
}
