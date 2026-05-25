## API 接口
### 健康检查 
GET http://ai-director.f1.luyouxia.net:15165/health (本地：http://localhost:8014/health)  
输出:
```json
{
    "status": "healthy",
    "service": "video-director",
    "timestamp": "2026-05-18T09:52:47.726329"
}
```
### 文案合规检查 
POST http://ai-director.f1.luyouxia.net:15165/check_compliance (本地：http://localhost:8014/check_compliance)  
输入：
```json
{
    "script": "文案内容"
}
```
输出：
```json
{
    "success": true,
    "data": {
        "compliant": false,
        "reason": "使用‘幽会’为不规范表达，属错别字替代‘优惠’，违反不规范表达红线"
    },
    "error": null,
    "error_code": null
}
```
### 文案优化
POST http://ai-director.f1.luyouxia.net:15165/optimize_script (本地：http://localhost:8014/optimize_script)  
输入：
```json
{
    "script": "文案内容"
}
```
输出：
```json
{
    "success": true,
    "data": {
        "script": "优化后的内容"
    },
    "error": null,
    "error_code": null
}
```
### AI选材
POST http://ai-director.f1.luyouxia.net:15165/select_material (本地：http://localhost:8014/select_material)    
输入：
```json
{
    "city_name": "展会城市",
    "show_title": "展会名称",
    "show_address": "展会地点",
    "show_time": "展会日期",
    "user_input": "2026北京国际汽车文化节9月16日-23日在首钢展览馆举办，各大品牌发送全新车型",
    "industry_name": "车展",
    "industry_id": 1,
    "org_id": 1
}
```
输出：
```json
{
    "success": true,
    "data": {
        "video_ids": [4, 204, 205, 324, 327, 323, 331],
        "template_ids": [145, 5, 29, 151, 148, 146, 153],
        "bgm_ids": [5, 404, 408, 403, 400, 406, 405],
        "voice_ids": [141, 145, 150, 184, 199, 201, 225],
        "scripts": [
            "2026{展会城市}{展会名称}，全新车型集中亮相！{展会开始日期}到{展会结束日期}，锁定{展会地点}。各大品牌携重磅新车登场，从智能电动到豪华燃油，一次看个够。现场还能体验首发互动装置和主题文化展区，感受汽车与艺术的碰撞。免费观展名额有限，提前预约更省心！",
            "想第一时间看到各大品牌的新车？2026{展会城市}{展会名称}来了！{展会开始日期}至{展会结束日期}，就在{展会地点}。全球车企齐聚，首发阵容强大，智能科技、绿色出行、经典复刻全都有。更有汽车文化市集、复古车展区等你打卡。别错过这场融合科技与文化的年度盛会！",
            "2026{展会城市}{展会名称}定档{展会开始日期}到{展会结束日期}，{展会地点}不见不散！各大品牌全新车型首次集中发布，涵盖新能源、高性能、智能座舱等前沿方向。现场不止有车，还有潮流音乐、创意改装和汽车历史长廊，打造沉浸式汽车文化体验。记得提前登记，轻松入场！",
            "车迷注意！2026{展会城市}{展会名称}将于{展会开始日期}至{展会结束日期}在{展会地点}盛大开启。各大品牌带来年度重磅新车，首发、首秀、首展齐上阵。现场设置品牌专属体验区、文化互动工坊和限量周边派发点，逛展也能收获满满。预约通道已开，速来锁定席位！",
            "2026{展会城市}{展会名称}，一场不止于车的盛宴！{展会开始日期}到{展会结束日期}，{展会地点}邀您共赴汽车文化之约。各大品牌全新车型首发亮相，搭配艺术装置、主题论坛与亲子互动区，全家都能玩得尽兴。观展免费，但需提前预约，名额充足，建议错峰出行！"
        ]
    },
    "error": null,
    "error_code": null
}
```
### AI推荐 
POST http://ai-director.f1.luyouxia.net:15165/recommend (本地：http://localhost:8014/recommend)  
输入：
```json
{
	"task_id": 22,
	"show_id": 33,
	"org_id": 1,
	"industry_id": 1,
	"task": {
		"task_desc": "生成短视频",
		"task_type": 1,
		"video_type": 1,
		"ai_director": "突出人多热闹，商业感强",
		"template_strategy": 2,
		"retry_type": 0,
		"video_count": 2,
		"city_name": "北京",
		"show_title": "国际汽车文化节",
		"show_address": "首钢会展中心",
		"show_time": "2026-05-01到05-03",
		"show_desc": null,
		"mult_source": [{
			"source_video_id": 2903,
			"type": 2949,
			"videos": [{
					"material_id": 211,
					"material_path": "https://www.pexels.com/zh-cn/download/video/36033073.mp4",
					"is_opening": 1
				},
				{
					"material_id": 330,
					"material_path": "https://www.pexels.com/zh-cn/download/video/8490544.mp4",
					"is_opening": 0
				}
			],
			"templates": {
				"material_id": 153,
				"material_path": "https://xiuxiu-pro.meitudata.com/posters/d708ecd87c7a32f4309144b2f57a4a97.jpeg"
			},
			"bgms": [{
				"material_id": 405,
				"material_path": "https://freepd.cn/api/music/576f726c642f4e6f72746875722e6d7033.mp3"
			}],
			"voices": {
				"material_id": 173,
				"robot_show_name": "唐僧"
			},
			"scripts": {
				"material_id": 1148,
				"script_content": "我实在是太喜欢北京国际汽车文化节了！"
			}
		}],
		"retry_source": [{
			"source_video_id": 290,
			"type": 2999,
			"videos": [{
					"material_id": 356,
					"material_path": "https://www.pexels.com/zh-cn/download/video/13929678.mp4",
					"is_opening": 1
				},
				{
					"material_id": 329,
					"material_path": "https://www.pexels.com/zh-cn/download/video/31319337.mp4",
					"is_opening": 0
				}
			],
			"templates": {
				"material_id": 152,
				"material_path": "https://xiuxiu-pro.meitudata.com/posters/3eb50419b73499bf738eef9a3d3b25c9.jpeg"
			},
			"bgms": [{
				"material_id": 407,
				"material_path": "https://freepd.cn/api/music/436f6d6564792f48656c6c6f21204d6120426162792e6d7033.mp3"
			}],
			"voices": {
				"material_id": 198,
				"robot_show_name": "小辉"
			},
			"scripts": {
				"material_id": 1198,
				"script_content": "北京国际汽车文化节真是太棒了！"
			}
		}]
	},
	"result": {
		"videos": [{
				"material_id": 324,
				"material_path": "https://www.pexels.com/zh-cn/download/video/35715727.mp4",
				"is_opening": 0
			},
			{
				"material_id": 4,
				"material_path": "https://www.pexels.com/zh-cn/download/video/33749020.mp4",
				"is_opening": 1
			}
		],
		"templates": [{
			"material_id": 145,
			"material_path": "https://xiuxiu-pro.meitudata.com/posters/b5ff9c7bdcd0800d735bb9cf27cf82a6.jpg"
		}],
		"bgms": [{
			"material_id": 400,
			"material_path": "https://freepd.cn/api/music/486f72726f722f4372656570792048616c6c6f772e6d7033.mp3"
		}],
		"voices": [{
			"material_id": 139,
			"robot_show_name": "小美"
		}],
		"scripts": [{
				"material_id": 1139,
				"script_content": "2026北京国际汽车文化节来了，6月12日-15日，就在首钢会展中心，百余款新车齐亮相，车模表演现场抽奖high翻天，快带上你的家人朋友来逛展吧！"
			},
			{
				"material_id": 1131,
				"script_content": "2026北京国际汽车文化节将在6月12日-15日登录首钢会展中心，大牌新车云集，车模表演现场抽奖氛围热烈，热爱汽车的你千万不要错过！"
			}
		]
	}
}
```
输出：
```json
{
    "success": true,
    "data": {
        "task_id": 22,
        "show_id": 33,
        "org_id": 1,
        "industry_id": 1,
        "results": [
            {
                "video_opening_id": {
                    "material_id": 356,
                    "material_path": "https://www.pexels.com/zh-cn/download/video/13929678.mp4"
                },
                "video_regular_ids": [
                    {
                        "material_id": 324,
                        "material_path": "https://www.pexels.com/zh-cn/download/video/35715727.mp4"
                    },
                    {
                        "material_id": 322,
                        "material_path": "https://www.pexels.com/zh-cn/download/video/33684356.mp4"
                    }
                ],
                "template_id": {
                    "material_id": 5,
                    "material_path": "https://imgs.design006.com/202204/Design006_8QGaeTCrEK.jpg"
                },
                "bgm_ids": {
                    "material_id": 5,
                    "material_path": "https://freepd.cn/api/music/53636f72696e672f416374696f6e20537472696b652e6d7033.mp3"
                },
                "voice_ids": {
                    "material_id": 141,
                    "material_path": null
                },
                "scripts": {
                    "material_id": 1131,
                    "script_content": "2026北京国际汽车文化节将在6月12日-15日登录首钢会展中心，大牌新车云集，车模表演现场抽奖氛围热烈，热爱汽车的你千万不要错过！"
                }
            },
            {
                "video_opening_id": {
                    "material_id": 208,
                    "material_path": "https://www.pexels.com/zh-cn/download/video/4965417.mp4"
                },
                "video_regular_ids": [
                    {
                        "material_id": 324,
                        "material_path": "https://www.pexels.com/zh-cn/download/video/35715727.mp4"
                    },
                    {
                        "material_id": 322,
                        "material_path": "https://www.pexels.com/zh-cn/download/video/33684356.mp4"
                    }
                ],
                "template_id": {
                    "material_id": 5,
                    "material_path": "https://imgs.design006.com/202204/Design006_8QGaeTCrEK.jpg"
                },
                "bgm_ids": {
                    "material_id": 5,
                    "material_path": "https://freepd.cn/api/music/53636f72696e672f416374696f6e20537472696b652e6d7033.mp3"
                },
                "voice_ids": {
                    "material_id": 141,
                    "material_path": null
                },
                "scripts": {
                    "material_id": 1139,
                    "script_content": "2026北京国际汽车文化节来了，6月12日-15日，就在首钢会展中心，百余款新车齐亮相，车模表演现场抽奖high翻天，快带上你的家人朋友来逛展吧！"
                }
            }
        ]
    },
    "error": null,
    "error_code": null
}
```