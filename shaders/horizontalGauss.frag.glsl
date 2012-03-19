uniform sampler2D bloomTex;

uniform float texDx;

varying vec2 texcoord;

float distance(vec2 tcoords, vec2 uv)
{
    float dx = tcoords.x - uv.x;
    float dy = tcoords.y - uv.y;
    return sqrt(dx*dx + dy*dy);
}

vec3 gaussianBlur(int n)
{
    vec3 sum = 0.0;
    float u = texcoord.x - n/2.0 * texDx;
    float totWeight = 0.0;
    for (int i = 0; i < n; ++i, u += texDx){
        float d = distance(texcoord, vec2(u,texcoord.y ));
        float weight = exp(-d*d);
        sum += weight*texture2D(bloomTex, vec2(u,texcoord.y ));
        totWeight += weight;
    }
    return sum / totWeight;
}

void main() {
	//First color buffer
	gl_FragData[1] = vec4(gaussianBlur(10),1);
}
