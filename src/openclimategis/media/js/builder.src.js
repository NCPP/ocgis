Ext.application({
    name: 'OpenClimateGIS Query Builder',
    launch: function() {
    /////////////////////////////////////////////////// Application Entry Point
        Ext.create('Ext.container.Viewport', { // Viewport
            layout: 'fit',
            items: [
                { // Map container
                    xtype: 'panel',                    
                    id: 'map-container',
                    title: 'Mapping Example',
                    layout: 'fit',
                    items: [
                        { // Google Maps instance
                            xtype: 'panel',
                            disabled: true,
                            hidden: true,
                            layout: 'fit',
                            id: 'map-panel',
                            listeners: {
                                afterrender: function() {
                                    var m = new google.maps.Map(document.getElementById('map-panel'), {
                                        center: new google.maps.LatLng(-34.397, 150.644),
                                        mapTypeId: google.maps.MapTypeId.ROADMAP,
                                        zoom: 8
                                        });
                                    }
                                }
                            } // eo map-panel
                        ] // eo items
                    } // eo map-container
                ] // eo items
            }); // eo Ext.create
        ////////////////////////////////////////////////////////////////////////
        } // eo launch()
    }); // eo Ext.application


